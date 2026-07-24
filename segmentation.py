from pathlib import Path
from src.rttm_utils import read_rttm

ground_truth_path = Path("data/ground_truths")
diarised_path = Path("data/lstm_2d_diarisation")

oversegmented = 0
undersegmented = 0
correct = 0

for video in ground_truth_path.glob("*.rttm"):
    ground_truth = read_rttm(video)
    diarised = read_rttm(diarised_path / video.name)

    ground_truth_len = len({seg["speaker"] for seg in ground_truth})
    diarised_len = len({seg["speaker"] for seg in diarised})

    if ground_truth_len > diarised_len:
        undersegmented += 1
    elif ground_truth_len < diarised_len:
        oversegmented += 1
    else:
        correct += 1

print(undersegmented)
print(oversegmented)
print(correct)
