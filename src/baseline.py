import os
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv, find_dotenv
from rttm_utils import evaluate_diarisation

audios_path = Path("data/audios")
diarised_path = Path("data/baselines")
output_csv_path = Path("data/baseline_results.csv")
ground_truth_path = Path("data/ground_truths")


def diarise(audio_path):
    output_file = (diarised_path / audio_path.name).with_suffix(".rttm")
    if output_file.exists():
        return

    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)

    with open(output_file, "w") as rttm:
        output.speaker_diarization.write_rttm(rttm)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    HUGGINGFACE_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=HUGGINGFACE_ACCESS_TOKEN
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    pipeline.to(device)

    Path(diarised_path).mkdir(parents=True, exist_ok=True)
    audios = audios_path.glob("*.wav")
    for i, audio in enumerate(audios):
        print(f"{i + 1}: Diarising {audio}")
        diarise(audio)

    evaluate_diarisation(ground_truth_path, diarised_path, output_csv_path)
