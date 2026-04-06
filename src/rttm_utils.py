import pandas as pd
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


# Takes MM:SS string and returns value in seconds
def convert_to_seconds(time_string):
    minutes, seconds = time_string.split(":")
    return 60 * int(minutes) + int(seconds)


# Takes data as an array of {start, end, speaker} segments
def save_to_rttm(data, video_id, output_path):
    with open(output_path, "w") as f:
        for seg in data:
            # RTTM Standard: SPEAKER <file> <chan> <start> <duration> <NA> <NA> <label> <NA> <NA>
            line = f"SPEAKER {video_id} 1 {seg['start']:.3f} {(seg['end'] - seg['start']):.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
            f.write(line)



# Compares two folders with corresponding rttm files and writes results to csv
def evaluate_diarisation(ground_truth_path, comparison_path, output_file_path):
    metric = DiarizationErrorRate(collar=0.250)
    results = []

    diarised_files = comparison_path.glob("*.rttm")
    for diarised_file in diarised_files:
        ground_truth_file = ground_truth_path / diarised_file.name
        if not ground_truth_file.exists():
            print(f"No ground truth found for {diarised_file}, skipping...")
            continue

        ref_dict = load_rttm(ground_truth_file)
        hyp_dict = load_rttm(diarised_file)

        if len(ref_dict) == 0 or len(hyp_dict) == 0:
            continue

        ref = ref_dict[diarised_file.stem]
        hyp = hyp_dict[diarised_file.stem]

        uem = ref.get_timeline().support()
        current_der = metric(ref, hyp, uem=uem, detailed=True)

        results.append(
            {
                "filename": diarised_file.stem,
                "total_speech": round(current_der["total"], 2),
                "confusion": round(current_der["confusion"], 2),
                "miss": round(current_der["missed detection"], 2),
                "false_alarm": round(current_der["false alarm"], 2),
                "der_percent": round(current_der["diarization error rate"] * 100, 2),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)

    global_der = abs(metric) * 100
    print(f"\nDataset DER: {global_der:.2f}%")

    report = metric[:]
    total_time = report["total"]
    print(f"Confusion:    {(report['confusion'] / total_time) * 100:.2f}%")
    print(f"Missed:       {(report['missed detection'] / total_time) * 100:.2f}%")
    print(f"False Alarm:  {(report['false alarm'] / total_time) * 100:.2f}%")
    print(f"\nDetailed report saved to: {output_file_path}")
