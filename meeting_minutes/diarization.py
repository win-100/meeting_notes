import gc
import os
import soundfile as sf
import torch
from .models import DiarizationSegment

def diarize(path, num_speakers=None):
    """Community-1 avec WAV préchargé : TorchCodec n'est pas utilisé."""
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN manquant : ajoutez-le dans .env après avoir accepté les conditions Hugging Face.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    if pipeline is None:
        raise RuntimeError("Accès refusé à Community-1 : vérifiez HF_TOKEN et les conditions Hugging Face.")
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        output = pipeline({"waveform": torch.from_numpy(samples.T.copy()), "sample_rate": rate}, num_speakers=num_speakers)
        diarization = getattr(output, "exclusive_speaker_diarization", output.speaker_diarization)
        return [DiarizationSegment(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
