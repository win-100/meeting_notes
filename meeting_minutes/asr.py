import gc
import re
import subprocess
import wave
from pathlib import Path

import torch

from .models import TranscriptSegment


CHUNK_SECONDS = 40
MIN_CHUNK_SECONDS = 10
SILENCE_SEARCH_SECONDS = 10
SILENCE_DURATION_SECONDS = 0.35
SILENCE_THRESHOLD_DB = -35


def _audio_duration(path):
    """Return the exact duration of the prepared PCM WAV file."""
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _silence_starts(path):
    """Return silence starts detected by ffmpeg, in seconds."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            f"silencedetect=n={SILENCE_THRESHOLD_DB}dB:d={SILENCE_DURATION_SECONDS}",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        float(value)
        for value in re.findall(r"silence_start: ([0-9.]+)", result.stderr)
    ]


def _chunk_boundaries(duration, silence_starts):
    """Choose sentence-friendly chunk boundaries without exceeding the limit."""
    boundaries = [0.0]
    start = 0.0

    while start + CHUNK_SECONDS < duration:
        maximum_end = start + CHUNK_SECONDS
        earliest_silence = start + max(
            MIN_CHUNK_SECONDS, CHUNK_SECONDS - SILENCE_SEARCH_SECONDS
        )
        candidates = [
            silence_start
            for silence_start in silence_starts
            if earliest_silence <= silence_start <= maximum_end
        ]
        end = candidates[-1] if candidates else maximum_end
        boundaries.append(end)
        start = end

    boundaries.append(duration)
    return list(zip(boundaries, boundaries[1:]))


def _split_on_silence(audio_path, chunk_directory):
    """Create fixed-maximum WAV chunks, preferring nearby silence boundaries."""
    chunks = _chunk_boundaries(
        _audio_duration(audio_path), _silence_starts(audio_path)
    )

    for index, (start, end) in enumerate(chunks):
        chunk_path = chunk_directory / f"chunk_{index:05d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-i",
                str(audio_path),
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            ],
            check=True,
            capture_output=True,
        )

    return chunks


def transcribe(path):
    """Transcribe an audio file locally with Parakeet TDT.

    The input is split into at-most-20-second WAV chunks to stay within 6 GB of
    VRAM. When possible, a nearby silence is used as the boundary.
    """
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    try:
        # Parakeet TDT/NeMo 3 mixes FP32 activations in its decoder, so
        # model.half() fails. Clear resident models before loading Parakeet.
        if device == "cuda":
            torch.cuda.empty_cache()

        model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").eval()

        # CUDA Graphs trigger an illegal memory access on a GTX 1660 SUPER
        # from the second chunk onward; the non-graph decoder is stable.
        with open_dict(model.cfg.decoding.greedy):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        model = model.to(device)

        audio_path = Path(path)
        chunk_directory = audio_path.parent / "asr_chunks"
        chunk_directory.mkdir(exist_ok=True)

        chunks_with_offsets = _split_on_silence(audio_path, chunk_directory)

        segments = []
        chunks = sorted(chunk_directory.glob("chunk_*.wav"))

        for chunk_path, (offset, end) in zip(chunks, chunks_with_offsets):
            with torch.inference_mode():
                result = model.transcribe([str(chunk_path)], timestamps=True)[0]

            timestamps = getattr(result, "timestamp", {}).get("segment", [])
            text = getattr(result, "text", "").strip()
            timestamped_segments = [
                TranscriptSegment(
                    offset + float(timestamp.get("start", 0)),
                    offset + float(timestamp.get("end", 0)),
                    timestamp.get("segment") or timestamp.get("text", ""),
                )
                for timestamp in timestamps
                if isinstance(timestamp, dict)
            ]

            if timestamped_segments:
                segments.extend(timestamped_segments)
            elif text:
                segments.append(
                    TranscriptSegment(offset, end, text)
                )

            if device == "cuda":
                torch.cuda.empty_cache()

        return segments
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
