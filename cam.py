import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from queue import Queue


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


def get_iou(bb1, bb2):
    left = max(bb1[0], bb2[0])
    right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    top = max(bb1[1], bb2[1])
    bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if right < left or bottom < top:
        return 0.0

    intersection_area = (right - left) * (bottom - top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    return intersection_area / float(bb1_area + bb2_area - intersection_area)


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

mp_model_path = "models/face_landmarker.task"
yunet_model_path = "models/face_detection_yunet_2023mar.onnx"
sface_model_path = "models/face_recognition_sface_2021dec.onnx"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (320, 320))
recogniser = cv2.FaceRecognizerSF.create(sface_model_path, "")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=mp_model_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

known_faces = {}
tracked_faces = {}

face_id_counter = 0
mouth_ratio_history = {}

video_path = "data/videos/4A8orzGYQRI_2.mp4"
fvs = FileVideoStream(video_path).start()

frame_count = 0
time_per_frame = 1000 / fvs.fps

scale = 0.25
width = int(fvs.width * scale)
height = int(fvs.height * scale)
detector.setInputSize((width, height))

with FaceLandmarker.create_from_options(options) as landmarker:
    start_time = int(time.time() * 1000)
    while fvs.more():
        frame = fvs.read()

        frame_start_time = int(time.time() * 1000)
        elapsed_time = int(frame_start_time - start_time)
        if elapsed_time > time_per_frame * frame_count:
            frame_count += 1
            continue

        frame_count += 1

        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, faces = detector.detect(resized_frame)

        current_frame_identities = {}

        if faces is not None:
            for i, face in enumerate(faces):
                # YuNet returns [x, y, w, h, ...]
                rect = list(map(int, face[:4]))

                identity = None
                # Use tracking instead of face embedding if possible
                iou_threshold = 0.5
                for face_id, data in tracked_faces.items():
                    if get_iou(rect, data) > iou_threshold:
                        print(f"Tracked {face_id}")
                        identity = face_id
                        break

                force_redo_face_id = frame_count % 30 == 0

                # Get face embedding
                if identity is None or force_redo_face_id:
                    aligned_face = recogniser.alignCrop(resized_frame, face)
                    feature = recogniser.feature(aligned_face)

                    # Check if embedding is already stored in known_faces
                    face_match_confidence_threshold = 0.363
                    for face_id, saved_feature in known_faces.items():
                        score = recogniser.match(
                            feature, saved_feature, cv2.FACE_RECOGNIZER_SF_FR_COSINE
                        )
                        if score > face_match_confidence_threshold:
                            print(f"Face matched {face_id}")
                            identity = face_id
                            break

                # If embedding not already saved, then add face to known_faces
                if identity is None:
                    identity = f"Person_{face_id_counter}"
                    print(f"New face {face_id_counter}")
                    face_id_counter += 1
                    known_faces[identity] = feature

                current_frame_identities[identity] = rect

                x, y, w, h = rect

                # Add a small margin (20%) to the crop for better landmarking context
                # margin_x, margin_y = int(w * 0.2), int(h * 0.2)
                margin_x, margin_y = 0, 0  # Test with margin disabled
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(width, x + w + margin_x), min(height, y + h + margin_y)

                face_crop = resized_frame[y1:y2, x1:x2]

                # Rescale points back to full-sized frame
                abs_x1 = int(x1 / scale)
                abs_y1 = int(y1 / scale)
                abs_x2 = int(x2 / scale)
                abs_y2 = int(y2 / scale)

                if face_crop.size == 0:
                    continue

                # cv2.imwrite(f"data/images/{timestamp_ms}_{i}.png", face_crop)

                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)

                result = landmarker.detect(mp_image)

                # Place landmarks on face
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    p_top = landmarks[13]
                    p_bottom = landmarks[14]
                    p_left = landmarks[61]
                    p_right = landmarks[291]

                    v_dist = np.sqrt((p_top.x - p_bottom.x) ** 2 + (p_top.y - p_bottom.y) ** 2)
                    h_dist = np.sqrt((p_left.x - p_right.x) ** 2 + (p_left.y - p_right.y) ** 2)
                    mar = v_dist / h_dist

                    # Track history to detect movement/variance
                    if identity not in mouth_ratio_history:
                        mouth_ratio_history[identity] = []
                    mouth_ratio_history[identity].append(mar)

                    if len(mouth_ratio_history[identity]) > 20:
                        mouth_ratio_history[identity].pop(0)

                    # If MAR variance is high, they are likely speaking
                    mouth_ratio_variance_threshold = 0.0010
                    is_speaking = (
                        np.var(mouth_ratio_history[identity]) > mouth_ratio_variance_threshold
                    )

                    for index in LIP_INDICES:
                        landmark = landmarks[index]

                        frame_x = int((x1 + landmark.x * face_crop.shape[1]) / scale)
                        frame_y = int((y1 + landmark.y * face_crop.shape[0]) / scale)

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
                else:
                    print("Landmarks unable to be placed on face!")

                border_colour = (0, 255, 0) if is_speaking else (0, 0, 255)
                frame = cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), border_colour, 3)

        tracked_faces = current_frame_identities

        cv2.imshow("Lip reading", frame)

        processing_time = int(time.time() * 1000 - frame_start_time)
        print(1000 / processing_time)
        if processing_time < time_per_frame:
            # Frames are being processed too quickly
            if cv2.waitKey(int(time_per_frame - processing_time)) & 0xFF == ord("q"):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

print(face_id_counter)
fvs.stop()
cv2.destroyAllWindows()
