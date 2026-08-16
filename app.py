import os
import uuid
import base64
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from meeting_minutes.alignment import (
    align_transcription_with_speakers,
    merge_consecutive,
)
from meeting_minutes.asr import ASR_ENGINES, TRANSCRIPTION_LANGUAGES, transcribe
from meeting_minutes.audio import prepare_audio
from meeting_minutes.browser_audio_uploader import video_audio_uploader
from meeting_minutes.diarization import diarize
from meeting_minutes.ollama_client import Ollama
from meeting_minutes.subtitles import transcript_as_srt


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MINUTES_MODEL = os.getenv("OLLAMA_MINUTES_MODEL", "qwen3.5:4b-q4_K_M")

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


def speaker_color(speaker, speakers):
    """Give each displayed speaker a stable, readable conversation color."""
    palette = [
        ("#DCEEFF", "#164A7A"),
        ("#E5F6E9", "#1D5A30"),
        ("#FCE8D5", "#7A3E12"),
        ("#F0E5FF", "#56307A"),
        ("#FBE1EC", "#7A2146"),
        ("#E1F4F4", "#15575A"),
    ]
    return palette[speakers.index(speaker) % len(palette)]

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

upload = st.file_uploader("Fichier audio", type=["mp3", "wav", "m4a"])
video_audio = video_audio_uploader(key="local_video_to_audio")

if video_audio and not upload:
    try:
        upload = {
            "name": video_audio["name"],
            "data": base64.b64decode(video_audio["data"], validate=True),
        }
        st.success(f"✓ Audio extrait localement : {upload['name']}")
    except (KeyError, ValueError) as error:
        st.error(f"Audio extrait par le navigateur invalide : {error}")
        upload = None
participants = st.text_area("Participants (un par ligne)").splitlines()
asr_engine_label = st.selectbox(
    "Moteur de transcription",
    options=list(ASR_ENGINES),
    help="Whisper Turbo permet de fixer la langue ; Parakeet la détecte automatiquement.",
)
asr_engine = ASR_ENGINES[asr_engine_label]
language_label = st.selectbox(
    "Langue de transcription",
    options=list(TRANSCRIPTION_LANGUAGES),
    index=1,
    disabled=asr_engine == "parakeet",
    help=(
        "Pour une réunion essentiellement en français, choisissez Français. "
        "Utilisez la détection automatique seulement si les langues sont réellement mélangées."
    ),
)
language = TRANSCRIPTION_LANGUAGES[language_label]
if asr_engine == "parakeet":
    st.caption("Parakeet TDT v3 détecte automatiquement la langue ; ce réglage ne lui est pas transmis.")
expected = st.checkbox(
    "Utiliser ce nombre comme nombre de locuteurs attendu",
    value=bool(participants),
)

if upload and st.button("Préparer, transcrire et diariser"):
    work = Path("work") / str(uuid.uuid4())
    work.mkdir(parents=True)

    source = work / (upload["name"] if isinstance(upload, dict) else upload.name)
    source.write_bytes(upload["data"] if isinstance(upload, dict) else upload.getbuffer())

    try:
        wav, mp3 = prepare_audio(source, work)
        st.success("✓ Audio préparé")

        Ollama(OLLAMA_BASE_URL).unload_all()

        raw = transcribe(wav, engine=asr_engine, language=language)
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
    st.download_button(
        "Télécharger le transcript diarisé (.txt)",
        data=transcript,
        file_name="transcript-diarise.txt",
        mime="text/plain",
    )
    st.download_button(
        "Télécharger les sous-titres VLC (.srt)",
        data=transcript_as_srt(segments, mappings),
        file_name="transcript-diarise.srt",
        mime="application/x-subrip",
        help="Dans VLC : Sous-titres > Ajouter un fichier de sous-titres.",
    )

    with st.expander("Transcript diarisé", expanded=True):
        st.caption("Les heures sont affichées au début de chaque prise de parole.")
        displayed_speakers = list(
            dict.fromkeys(
                mappings.get(segment.speaker_id) or segment.speaker_id or "Inconnu"
                for segment in segments
            )
        )
        last_speaker = object()
        dialogue_html = []
        for segment in segments:
            speaker = mappings.get(segment.speaker_id) or segment.speaker_id or "Inconnu"
            background, text_color = speaker_color(speaker, displayed_speakers)
            header = ""
            if segment.speaker_id != last_speaker:
                header = (
                    '<div style="margin: 9px 0 1px; font-size: 0.85rem; '
                    f'font-weight: 600; color: {text_color};">'
                    f"{escape(speaker)} · {format_timestamp(segment.start)}</div>"
                )
                last_speaker = segment.speaker_id
            dialogue_html.append(
                f"{header}"
                '<div style="margin: 0 0 2px; padding: 5px 9px; '
                f'border-radius: 6px; line-height: 1.35; background-color: {background}; '
                f'color: {text_color};">'
                f"{escape(segment.text)}</div>",
            )
        st.markdown("".join(dialogue_html), unsafe_allow_html=True)

    prompt = st.text_area(
        "Prompt du compte-rendu",
        Path("prompts/default_minutes.txt").read_text(),
    )

    if MINUTES_MODEL in models:
        st.caption(f"Modèle du compte-rendu : `{MINUTES_MODEL}` — reasoning désactivé.")
        if st.button("Générer le compte-rendu"):
            response = Ollama(OLLAMA_BASE_URL).generate(
                MINUTES_MODEL,
                f"{prompt}\n\nVoici la transcription diarizée :\n{transcript}",
                think=False,
            )
            st.markdown(response)
    elif models:
        st.error(
            f"Le modèle requis `{MINUTES_MODEL}` n'est pas installé dans Ollama. "
            "Installez-le ou définissez OLLAMA_MINUTES_MODEL avec son tag local exact."
        )
