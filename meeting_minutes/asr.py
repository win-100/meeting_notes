import gc
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import torch

from .models import TranscriptSegment, Word


CHUNK_SECONDS = 40
MIN_CHUNK_SECONDS = 10
SILENCE_SEARCH_SECONDS = 10
SILENCE_DURATION_SECONDS = 0.35
SILENCE_THRESHOLD_DB = -35

ASR_ENGINES = {
    "Whisper Turbo (recommandé)": "whisper_turbo",
    "Parakeet TDT v3": "parakeet",
}
TRANSCRIPTION_LANGUAGES = {
    "Détection automatique": None,
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
}


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


def _transcribe_parakeet(chunks):
    """Transcribe pre-cut WAV chunks locally with Parakeet TDT."""
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

        segments = []
        for chunk_path, offset, end in chunks:
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


def _transcribe_whisper_turbo(chunks, language):
    """Transcribe pre-cut WAV chunks with Whisper Turbo."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Whisper Turbo n'est pas installé. Exécutez « pip install -r requirements.txt »."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    model = None

    try:
        if device == "cuda":
            torch.cuda.empty_cache()

        model = WhisperModel(
            "large-v3-turbo", device=device, compute_type=compute_type
        )
        segments = []
        for chunk_path, offset, _ in chunks:
            whisper_segments, _ = model.transcribe(
                str(chunk_path),
                language=language,
                task="transcribe",
                beam_size=5,
                word_timestamps=True,
                # Whisper can decode a long trailing silence as text seen frequently
                # in subtitle training data (for example "Sous-titrage FR").  Let
                # Silero VAD discard non-speech before decoding it.  Keeping windows
                # independent also prevents such a hallucination being propagated to
                # the next window.
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 1000},
                condition_on_previous_text=False,
            )
            segments.extend(
                TranscriptSegment(
                    start=offset + segment.start,
                    end=offset + segment.end,
                    text=segment.text.strip(),
                    words=[
                        Word(
                            offset + word.start,
                            offset + word.end,
                            word.word.strip(),
                        )
                        for word in (segment.words or [])
                        if word.word.strip()
                    ],
                )
                for segment in whisper_segments
                if segment.text.strip()
            )
        return segments
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe(path, engine="whisper_turbo", language="fr"):
    """Transcribe a local audio file with the selected ASR engine.

    All engines receive the same at-most-40-second WAV chunks, with boundaries
    moved to nearby silences when possible. Whisper Turbo accepts a language
    hint; Parakeet TDT v3 detects languages automatically and ignores it.
    """
    if engine not in {"whisper_turbo", "parakeet"}:
        raise ValueError(f"Moteur de transcription inconnu : {engine}")

    with tempfile.TemporaryDirectory(prefix="meeting_notes_asr_") as directory:
        chunk_directory = Path(directory)
        boundaries = _split_on_silence(Path(path), chunk_directory)
        chunks = [
            (chunk_directory / f"chunk_{index:05d}.wav", start, end)
            for index, (start, end) in enumerate(boundaries)
        ]

        if engine == "whisper_turbo":
            return _transcribe_whisper_turbo(chunks, language)
        if engine == "parakeet":
            return _transcribe_parakeet(chunks)
