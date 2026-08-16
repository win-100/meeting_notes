import subprocess
from pathlib import Path


def prepare_audio(source, work):
    """Convert a meeting recording to mono 16 kHz WAV and an MP3 copy."""
    work_directory = Path(work)
    wav_path = work_directory / "audio.wav"
    mp3_path = work_directory / "audio.mp3"

    base_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]

    subprocess.run(
        base_command + ["-c:a", "pcm_s16le", str(wav_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)],
        check=True,
        capture_output=True,
    )

    return wav_path, mp3_path
