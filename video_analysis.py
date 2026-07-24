import os
import subprocess
import json
import numpy as np


def get_video_stats(folder_path):
    # Supported video extensions
    extensions = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv")
    durations = []

    print(f"Scanning folder: {folder_path}...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            filepath = os.path.join(folder_path, filename)

            # Run ffprobe to get duration in seconds
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filepath,
            ]

            try:
                result = subprocess.check_output(cmd).decode("utf-8").strip()
                durations.append(float(result))
            except Exception as e:
                print(f"Skipping {filename}: Could not read duration.")

    if not durations:
        print("No video files found.")
        return

    # Calculations
    total_seconds = sum(durations)
    mean_val = np.mean(durations)
    variance_val = np.var(durations)
    min = np.min(durations)
    max = np.max(durations)

    # Output results
    print("-" * 30)
    print(f"Total Videos: {len(durations)}")
    print(f"Total Length: {total_seconds:.2f}s ({total_seconds / 60:.2f} minutes)")
    print(f"Mean Length:  {mean_val:.2f}s")
    print(f"Variance:     {variance_val:.2f}s²")
    print(f"Std Dev:      {np.sqrt(variance_val):.2f}s")
    print(f"Min:          {min}s")
    print(f"Max:          {max}s")
    print("-" * 30)


# Change '.' to your folder path
get_video_stats("data/processed_videos")
