import os
import json

import yt_dlp

from rttm_utils import convert_to_seconds

path = "test_subset_100.json"
videos_path = "data/videos"


def save_video(video_id, video_data):
    url = video_data["video_url"]
    for i, segment in enumerate(video_data["segments"]):

        output_path = f"{videos_path}/{video_id}_{i}.mp4"
        if os.path.exists(output_path):
            continue
        print(f"Downloading {video_id}")

        start = convert_to_seconds(segment["start_time"])
        end = convert_to_seconds(segment["end_time"])
        ydl_opts = {
            "outtmpl": f"{videos_path}/{video_id}_{i}.%(ext)s",
            "merge_output_format": "mp4",
            "quiet": True,
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
            "force_keyframes_at_cuts": True,
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)


if __name__ == "__main__":
    with open(path) as f:
        input_json = json.load(f)
        for video_id, video_data in input_json.items():
            save_video(video_id, video_data)
