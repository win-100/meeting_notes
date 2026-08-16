import gc
import subprocess
from pathlib import Path

import torch

from .models import TranscriptSegment


CHUNK_SECONDS = 20


def transcribe(path):
    """Transcribe an audio file locally with Parakeet TDT.

    The input is split into fixed-size WAV chunks to stay within 6 GB of VRAM.
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

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-f",
                "segment",
                "-segment_time",
                str(CHUNK_SECONDS),
                "-c:a",
                "pcm_s16le",
                str(chunk_directory / "chunk_%05d.wav"),
            ],
            check=True,
            capture_output=True,
        )

        segments = []
        chunks = sorted(chunk_directory.glob("chunk_*.wav"))

        for index, chunk_path in enumerate(chunks):
            with torch.inference_mode():
                result = model.transcribe([str(chunk_path)], timestamps=True)[0]

            offset = index * CHUNK_SECONDS
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
                    TranscriptSegment(offset, offset + CHUNK_SECONDS, text)
                )

            if device == "cuda":
                torch.cuda.empty_cache()

        return segments
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
