# Takes MM:SS string and returns value in seconds
def convert_to_seconds(time_string):
    minutes, seconds = time_string.split(":")
    return 60 * int(minutes) + int(seconds)


# Takes data as an array of {start, duration, speaker} segments
def save_to_rttm(data, video_id, output_path):
    with open(output_path, "w") as f:
        for seg in data:
            # RTTM Standard: SPEAKER <file> <chan> <start> <duration> <NA> <NA> <label> <NA> <NA>
            line = f"SPEAKER {video_id} 1 {data['start']:.3f} {seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
            f.write(line)
