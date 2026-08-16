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


def format_timestamp(seconds):
    """Return an elapsed timestamp suitable for a meeting transcript."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def transcript_with_diarization(segments, speaker_names):
    """Build the portable, human-readable diarized transcript."""
    lines = []
    for segment in segments:
        speaker = speaker_names.get(segment.speaker_id) or segment.speaker_id or "Inconnu"
        lines.append(f"[{format_timestamp(segment.start)}] {speaker}: {segment.text}")
    return "\n".join(lines)


def speaker_ids(segments):
    """Keep speaker choices in a predictable order."""
    return sorted(
        {segment.speaker_id for segment in segments if segment.speaker_id},
        key=lambda value: (
            not value.removeprefix("SPEAKER_").isdigit(),
            int(value.removeprefix("SPEAKER_"))
            if value.removeprefix("SPEAKER_").isdigit()
            else value,
        ),
    )

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
        st.session_state["speaker_names"] = {}
    except Exception as error:
        st.error(str(error))

if "segments" in st.session_state:
    segments = st.session_state["segments"]
    available_names = ["Inconnu"] + [participant.strip() for participant in participants if participant.strip()]
    mappings = st.session_state.setdefault("speaker_names", {})

    st.subheader("Interlocuteurs")
    st.caption("Associez chaque identifiant de diarisation une seule fois. Ces choix restent modifiables.")
    mapping_columns = st.columns(min(3, max(1, len(speaker_ids(segments)))))
    for index, speaker_id in enumerate(speaker_ids(segments)):
        choices = available_names.copy()
        current_name = mappings.get(speaker_id, "Inconnu")
        if current_name not in choices:
            choices.append(current_name)
        mappings[speaker_id] = mapping_columns[index % len(mapping_columns)].selectbox(
            speaker_id,
            choices,
            index=choices.index(current_name),
            key=f"speaker_mapping_{speaker_id}",
        )

    for segment in segments:
        segment.speaker_name = mappings.get(segment.speaker_id, "Inconnu")

    transcript = transcript_with_diarization(segments, mappings)
    panel_open = st.toggle("Afficher le transcript diarisé", value=True)

    main_column, transcript_column = (
        st.columns([3, 2], gap="large") if panel_open else (st.container(), None)
    )

    with main_column:
        st.download_button(
            "Télécharger le transcript diarisé (.txt)",
            data=transcript,
            file_name="transcript-diarise.txt",
            mime="text/plain",
        )

        prompt = st.text_area(
            "Prompt du compte-rendu",
            Path("prompts/default_minutes.txt").read_text(),
        )

        if models:
            model = st.selectbox("Modèle Ollama", models)
            if st.button("Générer le compte-rendu"):
                response = Ollama(OLLAMA_BASE_URL).generate(
                    model,
                    f"{prompt}\n\nVoici la transcription diarizée :\n{transcript}",
                )
                st.markdown(response)

    if transcript_column is not None:
        with transcript_column:
            st.subheader("Transcript diarisé")
            st.caption("Les heures sont affichées au début de chaque prise de parole.")
            last_speaker = object()
            for segment in segments:
                speaker = mappings.get(segment.speaker_id) or segment.speaker_id or "Inconnu"
                if segment.speaker_id != last_speaker:
                    st.markdown(
                        f"**{speaker}** · {format_timestamp(segment.start)}"
                    )
                    last_speaker = segment.speaker_id
                st.chat_message("assistant", avatar="💬").write(segment.text)
