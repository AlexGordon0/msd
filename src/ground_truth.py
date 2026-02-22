import time
import json
from rttm_utils import save_to_rttm
from os import listdir
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

audios_path = Path("data/videos")
ground_truths_path = Path("data/ground_truths")


def get_ground_truth(filename):
    uploaded_file = client.files.upload(file=(audios_path / filename))

    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    prompt = """
    You have been provided with a video. Perform speech diarisation by segmenting speakers.

    Instructions:
        1. Segment each speaker accurately and capture every utterance, including fillers like "Yeah" and "Exactly".
        2. Handle Overlaps: If speakers overlap, create two separate entries, one for each speaker.
        3. Identification: Label each speaker as SPEAKER_00, SPEAKER_01, etc.
        4. Store start time as a string in MM:SS to 3 decimals places and duration as a float to 3 decimal places.

    Output Format:
        - Return a JSON array only.
        - Schema: [{"speaker": "ID", "start": "MM:SS", "duration": float}, ...]
        - Ensure "start" and "duration" are floats to 3 decimal places.
        - Sort strictly by 'start' time.
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    client.files.delete(name=uploaded_file.name)
    return json.loads(response.text)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    client = genai.Client()

    Path(ground_truths_path).mkdir(parents=True, exist_ok=True)
    audios = [f for f in listdir(audios_path)]

    for i, audio in enumerate(audios):
        rttm_filename = Path(audio).with_suffix(".rttm")
        out_rttm = Path(ground_truths_path) / f"{rttm_filename}"
        print(f"{i}: Getting ground truth for {Path(audio).stem}")
        if not out_rttm.exists():
            response = get_ground_truth(audio)
            save_to_rttm(response, Path(audio).stem, out_rttm)
