import os

import torch
from dotenv import find_dotenv, load_dotenv
from pyannote.audio import Pipeline
from pathlib import Path

from src.audio import diarise_from_array
from src.cam import run_pipeline
from src.extract_audios import extract_audio
from src.fusion_engine import merge_diarisation
from src.rttm_utils import save_to_rttm, evaluate_diarisation

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    HUGGINGFACE_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=HUGGINGFACE_ACCESS_TOKEN
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    pipeline.to(device)

    videos_path = Path("data/videos")
    output_folder = Path("data/combined_diarisation")
    output_folder.mkdir(parents=True, exist_ok=True)
    ground_truth_folder = Path("data/ground_truths")

    for filename in videos_path.glob("*.mp4"):
        audio, sample_rate = extract_audio(filename)
        audio_segments = diarise_from_array(audio, sample_rate, pipeline)
        video_segments = run_pipeline(filename)
        # print("-" * 80)
        # print(audio_segments)
        # print("-" * 80)
        # print(video_segments)
        # print("-" * 80)

        merged_output = merge_diarisation(audio_segments, video_segments)
        # print(merged_output)
        output_path = output_folder / filename.name.with_suffix(".rttm")
        save_to_rttm(merged_output, filename.stem, output_path)

    evaluate_diarisation(ground_truth_folder, output_folder, "data/combined_results.csv")
