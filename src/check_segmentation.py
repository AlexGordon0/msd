import re
from pathlib import Path


def get_speaker_set(file_path):
    speaker_set = set()
    pattern = r"SPEAKER_\d{2}"
    with open(file_path) as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match:
                speaker_set.add(match.group())
    return speaker_set

ground_truth_path = Path("data/ground_truths")
comparison_path = Path("data/diarised_video")

oversegmented, correctly_segmented, undersegmented = 0, 0, 0

for rttm in comparison_path.glob("*.rttm"):
    filename = rttm.name
    ground_truth_file = ground_truth_path / filename

    comparison_set = get_speaker_set(rttm)
    ground_truth_set = get_speaker_set(ground_truth_file)

    if len(comparison_set) > len(ground_truth_set):
        oversegmented += 1
    elif len(comparison_set) < len(ground_truth_set):
        undersegmented += 1
    else:
        correctly_segmented += 1

print(f"Over | Correct | Under: {oversegmented}, {correctly_segmented}, {undersegmented}")
