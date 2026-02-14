import cv2
import mediapipe as mp
import time


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

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_model_path = "face_landmarker.task"
yunet_model_path = "face_detection_yunet_2023mar.onnx"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=mp_model_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

detector = cv2.FaceDetectorYN.create(
    model=yunet_model_path,
    config="",
    input_size=(320, 320),
)

cap = cv2.VideoCapture("data/videos/4A8orzGYQRI_2.mp4")
if not cap.isOpened():
    print("COULD NOT OPEN CAPTURE")
    exit(1)

frame_count = 0
src_fps = cap.get(cv2.CAP_PROP_FPS)
time_per_frame = 1000 / src_fps

scale = 0.25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

detector.setInputSize((width, height))

with FaceLandmarker.create_from_options(options) as landmarker:
    start_time = int(time.time() * 1000)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = int(time.time() * 1000)
        elapsed_time = int(frame_start_time - start_time)
        if elapsed_time > time_per_frame * frame_count:
            frame_count += 1
            continue

        frame_count += 1

        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, faces = detector.detect(resized_frame)

        if faces is not None:
            for i, face in enumerate(faces):
                # YuNet returns [x, y, w, h, ...]
                x, y, w, h = map(int, face[:4])

                # Add a small margin (20%) to the crop for better landmarking context
                margin_x, margin_y = int(w * 0.2), int(h * 0.2)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = min(width, x + w + margin_x), min(height, y + h + margin_y)

                face_crop = resized_frame[y1:y2, x1:x2]

                # Rescale points back to full-sized frame
                abs_x1 = int(x1 / scale)
                abs_y1 = int(y1 / scale)
                abs_x2 = int(x2 / scale)
                abs_y2 = int(y2 / scale)

                frame = cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 3)
                if face_crop.size == 0:
                    continue

                # cv2.imwrite(f"data/images/{timestamp_ms}_{i}.png", face_crop)

                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)

                result = landmarker.detect(mp_image)

                if result.face_landmarks:
                    for face_landmarks in result.face_landmarks:
                        for index in LIP_INDICES:
                            landmark = face_landmarks[index]

                            frame_x = int((x1 + landmark.x * face_crop.shape[1]) / scale)
                            frame_y = int((y1 + landmark.y * face_crop.shape[0]) / scale)

                            frame = cv2.circle(
                                frame,
                                (
                                    frame_x,
                                    frame_y,
                                ),
                                1,
                                (255, 0, 0),
                                -1,
                            )

        cv2.imshow("Lip reading", frame)

        processing_time = int(time.time() * 1000 - frame_start_time)
        if processing_time < time_per_frame:
            # Frames are being processed too quickly
            if cv2.waitKey(int(time_per_frame - processing_time)) & 0xFF == ord("q"):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
