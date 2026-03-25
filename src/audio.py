import os
from pathlib import Path

import torch
from scipy.io import wavfile
from dotenv import find_dotenv, load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from src.rttm_utils import evaluate_diarisation, save_to_rttm

audios_path = Path("data/audios")
diarised_path = Path("data/baselines")
output_csv_path = Path("data/baseline_results.csv")
ground_truth_path = Path("data/ground_truths")


def diarise_from_array(waveform_np, sample_rate, pipeline):
    waveform_tensor = torch.from_numpy(waveform_np).float()

    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    waveform_tensor = waveform_tensor.T

    diarisation = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
    segments = diarisation.serialize()["diarization"]

    return segments


def diarise_to_file(audio_path, pipeline):
    output_file = (diarised_path / audio_path.name).with_suffix(".rttm")
    if output_file.exists():
        return

    sample_rate, audio_array = wavfile.read(audio_path)
    segments = diarise_from_array(audio_array, sample_rate, pipeline)

    save_to_rttm(segments, output_file.stem, output_file) 

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
        diarise_to_file(audio, pipeline)

    evaluate_diarisation(ground_truth_path, diarised_path, output_csv_path)
