from os import listdir
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip

videos_path = Path("data/videos")
audios_path = Path("data/audios")


def extract_audio(filename):
    video_path = videos_path / filename
    audio_filename = Path(filename).with_suffix(".wav")
    output_path = audios_path / audio_filename
    if output_path.exists():
        return

    with VideoFileClip(str(video_path)) as video:
        video.audio.write_audiofile(
            str(output_path), fps=16000, nbytes=2, codec="pcm_s16le", logger=None
        )


if __name__ == "__main__":
    Path(audios_path).mkdir(parents=True, exist_ok=True)
    videos = [f for f in listdir(videos_path)]
    for video in videos:
        extract_audio(video)
