import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from meeting_minutes.alignment import (
    align_transcription_with_speakers,
    merge_consecutive,
)
from meeting_minutes.asr import transcribe
from meeting_minutes.audio import prepare_audio
from meeting_minutes.diarization import diarize
from meeting_minutes.ollama_client import Ollama


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

st.set_page_config(page_title="Meeting Minutes")
st.title("Transcription locale de réunions")

cuda = False
try:
    import torch

    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
except Exception:
    gpu = "PyTorch indisponible"

st.info(f'GPU : {gpu} — CUDA : {"disponible" if cuda else "indisponible"}')

try:
    models = Ollama(OLLAMA_BASE_URL).models()
    st.success("Ollama : disponible")
except Exception:
    models = []
    st.warning("Ollama n'est pas accessible.")

upload = st.file_uploader(
    "Fichier de réunion",
    type=["mp3", "wav", "m4a", "mp4", "mkv"],
)
participants = st.text_area("Participants (un par ligne)").splitlines()
expected = st.checkbox(
    "Utiliser ce nombre comme nombre de locuteurs attendu",
    value=bool(participants),
)

if upload and st.button("Préparer, transcrire et diariser"):
    work = Path("work") / str(uuid.uuid4())
    work.mkdir(parents=True)

    source = work / upload.name
    source.write_bytes(upload.getbuffer())

    try:
        wav, mp3 = prepare_audio(source, work)
        st.success("✓ Audio préparé")

        Ollama(OLLAMA_BASE_URL).unload_all()

        raw = transcribe(wav)
        st.success("✓ Transcription terminée")

        dia = diarize(wav, len(participants) if expected else None)
        st.success("✓ Diarisation terminée")

        aligned_segments = align_transcription_with_speakers(raw, dia)
        st.session_state["segments"] = merge_consecutive(aligned_segments)
        st.session_state["work"] = work
    except Exception as error:
        st.error(str(error))

if "segments" in st.session_state:
    names = ["Inconnu"] + [participant for participant in participants if participant]

    for index, segment in enumerate(st.session_state.segments):
        segment.speaker_name = st.selectbox(
            f"{segment.speaker_id} — {segment.text[:80]}",
            names,
            key=index,
        )

    prompt = st.text_area(
        "Prompt du compte-rendu",
        Path("prompts/default_minutes.txt").read_text(),
    )

    if models and st.button("Générer le compte-rendu"):
        model = st.selectbox("Modèle Ollama", models)
        transcript = "\n".join(
            f"[{segment.speaker_name}] {segment.text}"
            for segment in st.session_state.segments
        )
        response = Ollama(OLLAMA_BASE_URL).generate(
            model,
            f"{prompt}\n\nVoici la transcription diarizée :\n{transcript}",
        )
        st.markdown(response)
