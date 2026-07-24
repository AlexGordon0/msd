from pathlib import Path

def has_overlapping_speech(file_path):
    video_overlap_time = 0
    segments = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # RTTM format: SPEAKER <file> <chnl> <start> <duration> ...
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                segments.append((start_time, end_time, duration))

    # Sort segments by start time
    segments.sort(key=lambda x: x[0])

    # Check for overlaps
    for i in range(len(segments)):
        current_end = segments[i][1]
        j = i + 1
        while j < len(segments) and (segments[j][0] < current_end):
            overlap_time = min(segments[j][1], current_end) - segments[j][0]
            video_overlap_time += overlap_time
            j += 1

    return video_overlap_time


if __name__ == "__main__":
    ground_truths = Path("data/ground_truths").glob("*.rttm")
    total_overlap_time = 0

    for ground_truth in ground_truths:
        total_overlap_time += has_overlapping_speech(ground_truth)
    print(total_overlap_time)
