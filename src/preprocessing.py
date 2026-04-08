import h5py
import numpy as np
import mediapipe as mp
from pathlib import Path

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

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=mp_landmarker_path,
    ),
    running_mode=VisionRunningMode.IMAGE,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(landmarker_options)


def get_normalised_mouth(detection_result):
    landmarks = detection_result.face_landmarks[0]

    matrix = detection_result.facial_transformation_matrixes[0].data
    matrix = np.array(matrix).reshape(4, 4)
    inv_matrix = np.linalg.inv(matrix)

    normalised_mouth = []
    for i in LIP_INDICES:
        landmark = landmarks[i]
        v = np.array([landmark.x, landmark.y, landmark.z, 1.0])

        v_norm = np.dot(inv_matrix, v)

        normalised_mouth.append(v_norm[:3])

    normalised_mouth = np.array(normalised_mouth)

    centre = normalised_mouth.mean(axis=0)
    normalised_mouth -= centre

    return normalised_mouth


def get_landmarks(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return np.zeros((len(LIP_INDICES), 3)), False

    normalised_mouth = get_normalised_mouth(result)
    return normalised_mouth, True


face_images_path = Path("data/faceImages_small.h5")
landmarkers_path = Path("data/lip_landmarks_3d.h5")

with h5py.File(face_images_path, "r") as rf, h5py.File(landmarkers_path, "w") as wf:
    for split in ["train", "test"]:
        x = rf[f"x_{split}"]
        y = rf[f"y_{split}"]

        wf_x = wf.create_dataset(
            f"x_{split}", (len(x), 38, len(LIP_INDICES), 3), dtype="float32"
        )
        wf_y = wf.create_dataset(f"y_{split}", data=y)

        for i in range(len(x)):
            video = x[i]
            clip_landmarks = []
            success_count = 0

            for frame_idx in range(38):
                frame = video[frame_idx]
                landmark_coords, success = get_landmarks(frame)
                if success:
                    success_count += 1

                clip_landmarks.append(landmark_coords)

            print(f"{success_count / 38 * 100}%")

            wf_x[i] = np.array(clip_landmarks)
