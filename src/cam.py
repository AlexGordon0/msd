import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

from rttm_utils import evaluate_diarisation, save_to_rttm


class FileVideoStream:
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

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetector = mp.tasks.vision.FaceDetector

# detector_options =

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=mp_landmarker_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)


# Takes two bounding boxes of the format [x, y, width, height] and returns true if their IoU
# surpasses the threshold value.
def check_tracking(bb1, bb2, iou_threshold=0.5):
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
    if score > iou_threshold:
        return True

    return False


# Returns if cosine distance of faces is larger than threshold value
def check_face_embedding(face1, face2, threshold=0.363):
    score = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
    if score > threshold:
        return True

    return False


def get_bounding_box(landmarks, bbox):
    xs = [l.x for l in landmarks]
    ys = [l.y for l in landmarks]

    w, h = bbox[2], bbox[3]

    x1 = bbox[0] + w * min(xs)
    y1 = bbox[1] + h * min(ys)
    x2 = bbox[0] + w * max(xs)
    y2 = bbox[1] + h * max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def add_segment(face_memory, segments, identity, frame_timestamp):
    segment_start_time = face_memory[identity]["start_time"]
    duration = frame_timestamp - segment_start_time
    segments.append(
        {
            "start": segment_start_time / 1000,
            "speaker": identity,
            "duration": duration / 1000,
        }
    )


def update_speech_status(
    landmarks, identity, history_dict, face_mem, segments, timestamp, fps
):
    p_top, p_bottom = landmarks[13], landmarks[14]
    p_left, p_right = landmarks[61], landmarks[291]

    v_dist = np.linalg.norm((p_top.x - p_bottom.x, p_top.y - p_bottom.y))
    h_dist = np.linalg.norm((p_left.x - p_right.x, p_left.y - p_right.y))
    mar = v_dist / h_dist

    history = history_dict.setdefault(identity, [])
    history.append(mar)
    if len(history) > int(fps):
        history.pop(0)

    # If MAR variance is high, they are likely speaking
    mouth_ratio_variance_threshold = 0.0015
    is_speaking = np.var(history) > mouth_ratio_variance_threshold

    was_speaking = face_mem[identity]["is_speaking"]
    if is_speaking and not was_speaking:
        # The person has just started speaking
        face_mem[identity]["start_time"] = timestamp
    elif not is_speaking and was_speaking:
        # Person has just finished speaking
        add_segment(face_mem, segments, identity, timestamp)

    face_mem[identity]["is_speaking"] = is_speaking
    return is_speaking


def draw_face_info(frame, landmarks, bbox, is_speaking, scale):
    x, y, w, h = bbox

    # Rescale points back to full-sized frame
    abs_x1 = int(x / scale)
    abs_y1 = int(y / scale)
    abs_x2 = int((x + w) / scale)
    abs_y2 = int((y + h) / scale)

    for index in LIP_INDICES:
        landmark = landmarks[index]

        frame_x = int(abs_x1 + landmark.x * w / scale)
        frame_y = int(abs_y1 + landmark.y * h / scale)

        landmark_colour = (255, 0, 0)

        frame = cv2.circle(
            frame,
            (
                frame_x,
                frame_y,
            ),
            1,
            landmark_colour,
            -1,
        )

        border_colour = (0, 255, 0) if is_speaking else (0, 0, 255)
        frame = cv2.rectangle(
            frame,
            (abs_x1, abs_y1),
            (abs_x2, abs_y2),
            border_colour,
            3,
        )


def run_pipeline(video_path):
    face_memory = {}  # {id: {"bbox": [], "embedding": [], "is_speaking": False, start_time: 0.0}, ..., }
    mouth_ratio_history = {}  # Stores mouth ratio history for each speaker
    segments = []
    face_id_counter = 0

    scale = 0.25
    detection_interval = 30

    fvs = FileVideoStream(video_path).start()
    fps = fvs.fps
    time_per_frame = 1000 / fvs.fps

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        frame_count = 0
        start_time = int(time.time() * 1000)

        while fvs.more():
            frame = fvs.read()

            frame_start_time = int(time.time() * 1000)
            frame_timestamp = time_per_frame * frame_count
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Skip frames for real-time diarisation
            # elapsed_time = int(frame_start_time - start_time)
            # if elapsed_time > frame_timestamp:
            #     frame_count += 1
            #     continue

            if frame_count % detection_interval == 0:
                # Heavy detection logic runs on key frames
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
                        if check_tracking(data["bbox"], rect):
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
                            if check_face_embedding(data["embedding"], feature):
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
                        }
                    else:
                        face_memory[identity]["bbox"] = rect

            for identity, data in face_memory.items():
                prev_rect = data["bbox"]

                # Add margin to allow for movement between frames
                margin = 0.1
                x_margin = prev_rect[2] * margin
                y_margin = prev_rect[3] * margin

                x1 = max(0, int(prev_rect[0] - x_margin))
                y1 = max(0, int(prev_rect[1] - y_margin))
                x2 = min(
                    int(fvs.width * scale),
                    int(prev_rect[0] + prev_rect[2] + x_margin),
                )
                y2 = min(
                    int(fvs.height * scale),
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
                    landmarks = result.face_landmarks[0]
                    new_rect = get_bounding_box(landmarks, expanded_box)

                    # Calculate MAR and Update segments
                    is_speaking = update_speech_status(
                        landmarks,
                        identity,
                        mouth_ratio_history,
                        face_memory,
                        segments,
                        frame_timestamp,
                        fps,
                    )

                    draw_face_info(frame, landmarks, expanded_box, is_speaking, scale)

                    # Update the bounding box for the next frame
                    data["bbox"] = new_rect
                else:
                    print(f"Cannot place face landmarks on {identity}")

            cv2.imshow("Lip reading", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # processing_time = int(time.time() * 1000 - frame_start_time)
            # if processing_time < time_per_frame:
            #     # Frames are being processed too quickly
            #     if cv2.waitKey(int(time_per_frame - processing_time)) & 0xFF == ord(
            #         "q"
            #     ):
            #         break
            #     else:
            #         if cv2.waitKey(1) & 0xFF == ord("q"):
            #             break

    # Add entries for faces still speaking at end of video
    for identity, data in face_memory.items():
        if data["is_speaking"]:
            add_segment(face_memory, segments, identity, frame_timestamp)

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
        video = Path("data/videos/p78K12jx_o4_0.mp4")
        output_path = diarised_path / (video.with_suffix(".rttm").name)
        print(f"{i + 1}: Running pipeline for {video.stem}")
        # if output_path.exists():
        # continue
        segments = run_pipeline(video)
        print(segments)
        save_to_rttm(segments, video.stem, output_path)
        break
    evaluate_diarisation(ground_truth_path, diarised_path, output_csv_path)
