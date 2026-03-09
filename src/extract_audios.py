import os
import tempfile
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.io import wavfile

videos_path = Path("data/videos")
audios_path = Path("data/audios")


def extract_audio(filename, sample_rate=16000):
    video = VideoFileClip(str(filename))

    if video.audio is None:
        print("No audio track found in this video.")
        return None, sample_rate

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")

    try:
        video.audio.write_audiofile(
            temp_path, fps=sample_rate, nbytes=2, codec="pcm_s16le", logger=None
        )

        rate, audio_array = wavfile.read(temp_path)

        # Convert to float32 if you need it normalized (-1 to 1)
        # audio_array = audio_array.astype(np.float32) / 32768.0

    finally:
        video.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return audio_array, sample_rate


def save_audio(filename):
    audio_filename = Path(filename).with_suffix(".wav").name
    output_path = audios_path / audio_filename
    if output_path.exists():
        return

    with VideoFileClip(str(filename)) as video:
        video.audio.write_audiofile(
            str(output_path), fps=16000, nbytes=2, codec="pcm_s16le", logger=None
        )


if __name__ == "__main__":
    Path(audios_path).mkdir(parents=True, exist_ok=True)
    videos = videos_path.glob("*.mp4")
    for video in videos:
        save_audio(video)
