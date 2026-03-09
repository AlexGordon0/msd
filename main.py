import os
import sys

import torch
from dotenv import find_dotenv, load_dotenv
from pyannote.audio import Pipeline

from src.audio import diarise_from_array
from src.cam import run_pipeline
from src.extract_audios import extract_audio

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py [video_file_path]")
        sys.exit(1)
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"Could not find file: {filename}")
        sys.exit(1)

    load_dotenv(find_dotenv())
    HUGGINGFACE_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=HUGGINGFACE_ACCESS_TOKEN
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    pipeline.to(device)

    audio, sample_rate = extract_audio(filename)
    audio_segments = diarise_from_array(audio, sample_rate, pipeline)
    print(audio_segments)
    # video_segments = run_pipeline(filename)
