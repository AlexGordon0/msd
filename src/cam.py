import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import torch
import mediapipe as mp
import numpy as np
from deepface import DeepFace

from src.rttm_utils import evaluate_diarisation, save_to_rttm
from src.preprocessing import get_normalised_mouth
from src.train_nn import SpeechDetectorTransformer


class FileVideoStream:
    """
    Video manager to perform asynchronous frame reading
    """

    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.01)  # Wait for consumer

    def read(self):
        return self.queue.get()

    def more(self):
        return self.queue.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()


LIP_INDICES = [
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    37,
    38,
    39,
    40,
    41,
    42,
    61,
    62,
    72,
    73,
    74,
    76,
    77,
    78,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    95,
    96,
    146,
    178,
    179,
    180,
    181,
    183,
    184,
    185,
    191,
    267,
    268,
    269,
    270,
    271,
    272,
    291,
    292,
    302,
    303,
    304,
    306,
    307,
    308,
    310,
    311,
    312,
    314,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    324,
    325,
    375,
    402,
    403,
    404,
    405,
    407,
    408,
    409,
    415,
]

mp_landmarker_path = "models/face_landmarker.task"
speech_model_path = "models/speech_detector_simple_200_epochs.pth"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=mp_landmarker_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)


def _check_tracking(bb1, bb2, iou_threshold=0.5):
    """
    Takes two bounding boxes of the format [x, y, width, height] and returns
    true if their IoU surpasses the threshold value.
    """
    left = max(bb1[0], bb2[0])
    right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    top = max(bb1[1], bb2[1])
    bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if right < left or bottom < top:
        return False

    intersection_area = (right - left) * (bottom - top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    score = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return score > iou_threshold


def _check_face_embedding(face1, face2, threshold=0.363):
    """
    Returns True if cosine distance of faces is larger than threshold value, else returns False
    """
    score = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
    return score > threshold


def _get_bounding_box(landmarks, bbox):
    """
    Calculate new bounding box for face based off of the most extreme landmark positions
    """
    xs = [l.x for l in landmarks]
    ys = [l.y for l in landmarks]

    w, h = bbox[2], bbox[3]

    x1 = bbox[0] + w * min(xs)
    y1 = bbox[1] + h * min(ys)
    x2 = bbox[0] + w * max(xs)
    y2 = bbox[1] + h * max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _add_segment(face_memory, segments, identity, frame_timestamp, confidence=1):
    """
    Add new segment to given segments array
    """
    if confidence < 0 or confidence > 1:
        print(f"Invalid confidence value: {confidence}")
        return
    segment_start_time = face_memory[identity]["start_time"]
    segments.append(
        {
            "start": round(segment_start_time / 1000, 3),
            "end": round(frame_timestamp / 1000, 3),
            "speaker": identity,
            "confidence": confidence,
        }
    )


def _normalise_landmarks(np_landmarks):
    centre = np_landmarks.mean(axis=0)
    np_landmarks -= centre
    return np_landmarks


def _update_speech_status(model, landmarks_data):
    """
    Update the speaking status of the speaker based off of landmark history
    """

    landmarks_tensor = torch.tensor(landmarks_data, dtype=torch.float32)
    landmarks_tensor = landmarks_tensor.unsqueeze(0)

    with torch.no_grad():
        raw_logits = model(landmarks_tensor)

        probability = torch.sigmoid(raw_logits)
        is_speaking = probability > 0.5
        return is_speaking, probability


def _draw_face_info(frame, landmarks, mean, bbox, is_speaking, confidence, SCALE):
    """
    Draw list of landmarks and bounding box onto the given frame
    If is_speaking is True, then bounding box is green, otherwise red
    """
    x, y, w, h = bbox

    # Rescale points back to full-sized frame
    abs_x1 = int(x / SCALE)
    abs_y1 = int(y / SCALE)
    abs_x2 = int((x + w) / SCALE)
    abs_y2 = int((y + h) / SCALE)

    landmark_colour = (255, 0, 0)
    for i in LIP_INDICES:
        landmark = landmarks[i]

        frame_x = int(abs_x1 + landmark.x * w / SCALE)
        frame_y = int(abs_y1 + landmark.y * h / SCALE)

        cv2.circle(
            frame,
            (
                frame_x,
                frame_y,
            ),
            1,
            landmark_colour,
            -1,
        )

    mean_x = int(abs_x1 + mean[0] * w / SCALE)
    mean_y = int(abs_y1 + mean[1] * h / SCALE)

    cv2.circle(
        frame,
        (
            mean_x,
            mean_y,
        ),
        5,
        (0, 0, 255),
        -1,
    )

    border_colour = (0, 255, 0) if is_speaking else (0, 0, 255)
    cv2.rectangle(
        frame,
        (abs_x1, abs_y1),
        (abs_x2, abs_y2),
        border_colour,
        3,
    )


def run_pipeline(video_path, display_output=False):
    # {id: {"bbox": [], "embedding": [], "is_speaking": False, "start_time": 0.0, "landmarks_memory": (38, 80, 2) array}, ..., }
    face_memory = {}
    face_id_counter = 0
    missed_frame_count = {}
    segments = []

    SCALE = 0.25

    fvs = FileVideoStream(video_path).start()
    fps = fvs.fps
    time_per_frame = 1000 / fps

    model = SpeechDetectorTransformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(
            speech_model_path, map_location=torch.device(device), weights_only=True
        )
    )
    model.eval()

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        frame_count = 0

        force_recalibration = True
        while fvs.more():
            frame = fvs.read()

            frame_timestamp = int(time_per_frame * frame_count)
            resized_frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

            if force_recalibration:
                force_recalibration = False

                for identity, data in missed_frame_count.items():
                    if data != 0:
                        # Face has been hidden from frame
                        # Add segment with 0 confidence
                        _add_segment(
                            face_memory, segments, identity, frame_timestamp, 0
                        )

                missed_frame_count = {}

                faces = DeepFace.extract_faces(
                    img_path=resized_frame,
                    detector_backend="retinaface",
                    enforce_detection=False,
                )

                # Update face memory with new bounding box locations
                for i, face_data in enumerate(faces):
                    area = face_data["facial_area"]
                    rect = [
                        area["x"],
                        area["y"],
                        area["w"],
                        area["h"],
                    ]

                    identity = None

                    # Use tracking instead of face embedding if possible
                    for id, data in face_memory.items():
                        if _check_tracking(data["bbox"], rect):
                            identity = id
                            break

                    # Get face embedding
                    if identity is None:
                        embedding_objs = DeepFace.represent(
                            face_data["face"],
                            model_name="ArcFace",
                            enforce_detection=False,
                            detector_backend="skip",
                            align=True,
                        )
                        feature = np.array(embedding_objs[0]["embedding"])

                        for id, data in face_memory.items():
                            if _check_face_embedding(data["embedding"], feature):
                                identity = id
                                break

                    # If face still not known, create new speaker
                    if identity is None:
                        identity = f"SPEAKER_{face_id_counter:02d}"
                        print(f"New face {identity}")
                        face_id_counter += 1
                        face_memory[identity] = {
                            "embedding": feature,
                            "bbox": rect,
                            "is_speaking": False,
                            "start_time": 0.0,
                            "landmarks_memory": [],
                        }
                    else:
                        face_memory[identity]["bbox"] = rect

                    missed_frame_count[identity] = 0

            for identity, data in face_memory.items():
                prev_rect = data["bbox"]

                # Add margin to allow for movement between frames
                margin = 0.1
                x_margin = prev_rect[2] * margin
                y_margin = prev_rect[3] * margin

                x1 = max(0, int(prev_rect[0] - x_margin))
                y1 = max(0, int(prev_rect[1] - y_margin))
                x2 = min(
                    int(fvs.width * SCALE),
                    int(prev_rect[0] + prev_rect[2] + x_margin),
                )
                y2 = min(
                    int(fvs.height * SCALE),
                    int(prev_rect[1] + prev_rect[3] + y_margin),
                )

                face_crop = resized_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=face_crop_rgb
                )
                result = landmarker.detect(mp_image)

                expanded_box = [x1, y1, x2 - x1, y2 - y1]

                if result.face_landmarks:
                    if (
                        missed_frame_count.get(identity, 0) != 0
                        and face_memory[identity]["is_speaking"]
                    ):
                        # Face has been hidden from frame, add segment with 0 confidence
                        _add_segment(
                            face_memory, segments, identity, frame_timestamp, 0
                        )
                        face_memory[identity]["start_time"] = frame_timestamp
                    missed_frame_count[identity] = 0

                    normalised_landmarks = get_normalised_mouth(result)
                    landmarks = result.face_landmarks[0]

                    lip_landmarks = np.array(
                        [[landmarks[i].x, landmarks[i].y] for i in LIP_INDICES]
                    )
                    # print(lip_landmarks)
                    # print(lip_landmarks.mean(axis=0))

                    new_rect = _get_bounding_box(landmarks, expanded_box)

                    landmarks_memory = face_memory[identity]["landmarks_memory"]
                    landmarks_memory.append(normalised_landmarks)
                    if len(landmarks_memory) > 38:
                        landmarks_memory.pop(0)

                    np_landmarks_memory = np.array(landmarks_memory)

                    # Predict speaking status using neural network
                    if len(np_landmarks_memory) == 38:
                        is_speaking, confidence = _update_speech_status(
                            model, np_landmarks_memory
                        )
                    else:
                        is_speaking, confidence = False, 0.0

                    # if identity == "SPEAKER_02":
                    #     print(f"{identity}: {is_speaking}, {confidence}")
                    #     print(f"{lip_landmarks}")
                    #     print(f"{identity} mean: {lip_landmarks.mean(axis=0)}")

                    was_speaking = face_memory[identity]["is_speaking"]
                    if is_speaking and not was_speaking:
                        # The person has just started speaking
                        face_memory[identity]["start_time"] = frame_timestamp
                    elif not is_speaking and was_speaking:
                        # Person has just finished speaking
                        _add_segment(
                            face_memory, segments, identity, frame_timestamp, confidence
                        )

                    face_memory[identity]["is_speaking"] = is_speaking

                    if display_output:
                        _draw_face_info(
                            frame,
                            landmarks,
                            lip_landmarks.mean(axis=0),
                            expanded_box,
                            is_speaking,
                            confidence,
                            SCALE,
                        )

                    # Update the bounding box for the next frame
                    data["bbox"] = new_rect
                else:
                    # Face cannot be found by mediapipe
                    if identity in missed_frame_count:
                        print(f"{frame_timestamp}: Missed frame for {identity}")
                        missed_frame_count[identity] += 1

                        # If they are speaking and a frame is missed then start new segment with new confidence value
                        if (
                            face_memory[identity]["is_speaking"]
                            and missed_frame_count[identity] == 1
                        ):
                            _add_segment(
                                face_memory, segments, identity, frame_timestamp
                            )
                            face_memory[identity]["start_time"] = frame_timestamp

                        # Allow 1 second of missed tracking before heavy recalibration is forced
                        if missed_frame_count[identity] >= fps:
                            force_recalibration = True

            if display_output:
                cv2.imshow("Lip reading", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Add entries for faces still speaking at end of video
    for identity, data in face_memory.items():
        if data["is_speaking"]:
            _add_segment(face_memory, segments, identity, frame_timestamp)

    fvs.stop()
    cv2.destroyAllWindows()
    return sorted(segments, key=lambda x: x["start"])


if __name__ == "__main__":
    videos_path = Path("data/videos")
    ground_truth_path = Path("data/ground_truths")
    output_csv_path = Path("data/video_only_results.csv")
    diarised_path = Path("data/diarised_video")
    diarised_path.mkdir(parents=True, exist_ok=True)

    videos = Path("data/videos").glob("*.mp4")
    for i, video in enumerate(videos):
        video = Path("data/videos/4A8orzGYQRI_2.mp4") # Remove if doing full video baseline
        output_path = diarised_path / (video.with_suffix(".rttm").name)
        print(f"{i + 1}: Running pipeline for {video.stem}")
        segments = run_pipeline(video, True)
        print(segments)
        save_to_rttm(segments, video.stem, output_path)
        break
    evaluate_diarisation(ground_truth_path, diarised_path, output_csv_path)
