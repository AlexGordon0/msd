import os
import sys

import torch
from dotenv import find_dotenv, load_dotenv
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from pathlib import Path

from src.audio import diarise_from_array
from src.cam import run_pipeline
from src.extract_audios import extract_audio
from src.fusion_engine import merge_diarisation
from src.rttm_utils import save_to_rttm
from src.train_nn import SpeechDetectorNN

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

    model = SpeechDetectorNN()
    speech_model_path = "models/speech_detector_lstm_2d.pth"
    model.load_state_dict(
        torch.load(
            speech_model_path, map_location=torch.device(device), weights_only=True
        )
    )
    model.eval()


    audio, sample_rate = extract_audio(filename)
    audio_segments = diarise_from_array(audio, sample_rate, pipeline)
    video_segments = run_pipeline(filename, model, True)

    print("-" * 80)
    print(audio_segments)
    print("-" * 80)
    print(video_segments)
    print("-" * 80)

    merged_output = merge_diarisation(audio_segments, video_segments)
    print(merged_output)
    filename_path = Path(filename)
    output_path = filename_path.with_suffix(".rttm").name
    save_to_rttm(merged_output, Path(filename).stem, output_path)

    metric = DiarizationErrorRate(collar=0.250)
    ref_dict = load_rttm(
        Path("data/ground_truths") / filename_path.with_suffix(".rttm").name
    )
    hyp_dict = load_rttm(output_path)

    if len(ref_dict) == 0 or len(hyp_dict) == 0:
        exit(0)

    ref = ref_dict[Path(output_path).stem]
    hyp = hyp_dict[Path(output_path).stem]

    uem = ref.get_timeline().support()
    current_der = metric(ref, hyp, uem=uem, detailed=True)
    report = metric[:]
    print(report.keys())
    total_time = report["total"]
    missed = report["missed detection"]
    false_alarm = report["false alarm"]
    confusion = report["confusion"]
    print(f"Missed:       {(missed / total_time) * 100:.2f}%")
    print(f"False Alarm:  {(false_alarm / total_time) * 100:.2f}%")
    print(f"Confusion:    {(confusion / total_time) * 100:.2f}%")
    print(f"DER:          {((missed + false_alarm + confusion) / total_time) * 100:.2f}%")
