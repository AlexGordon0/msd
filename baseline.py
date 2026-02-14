import os
import torch
import pandas as pd
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from dotenv import load_dotenv, find_dotenv

audios_path = Path("data/audios")
diarised_path = Path("data/baselines")
ground_truth_path = Path("data/ground_truths")


def diarise(audio_path):
    output_file = (diarised_path / audio_path.name).with_suffix(".rttm")
    if output_file.exists():
        return

    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)

    with open(output_file, "w") as rttm:
        output.speaker_diarization.write_rttm(rttm)


def evaluate_diarisation():
    metric = DiarizationErrorRate(collar=0.250)
    results = []
    output_csv = Path("data/diarisation_results.csv")

    diarised_files = diarised_path.glob("*.rttm")
    for diarised_file in diarised_files:
        ground_truth_file = ground_truth_path / diarised_file.name
        if not ground_truth_file.exists():
            print(f"No ground truth found for {diarised_file}, skipping...")
            continue

        ref_dict = load_rttm(ground_truth_file)
        hyp_dict = load_rttm(diarised_file)

        ref = ref_dict[diarised_file.stem]
        hyp = hyp_dict[diarised_file.stem]

        uem = ref.get_timeline().support()
        current_der = metric(ref, hyp, uem=uem, detailed=True)

        results.append(
            {
                "filename": diarised_file.name,
                "total_speech": round(current_der["total"], 2),
                "confusion": round(current_der["confusion"], 2),
                "miss": round(current_der["missed detection"], 2),
                "false_alarm": round(current_der["false alarm"], 2),
                "der_percent": round(current_der["diarization error rate"] * 100, 2),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    global_der = abs(metric) * 100
    print(f"\nDataset DER: {global_der:.2f}%")

    report = metric[:]
    total_time = report["total"]
    print(f"Confusion:    {(report['confusion'] / total_time) * 100:.2f}%")
    print(f"Missed:       {(report['missed detection'] / total_time) * 100:.2f}%")
    print(f"False Alarm:  {(report['false alarm'] / total_time) * 100:.2f}%")
    print(f"\nDetailed report saved to: {output_csv}")


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

    evaluate_diarisation()
