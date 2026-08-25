import os
import uuid
import base64
import json
from io import BytesIO
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

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
from meeting_minutes.models import dump
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


def speaker_label(speaker_id, speaker_names):
    """Return the selected speaker name, or its diarization ID as a fallback."""
    selected_name = speaker_names.get(speaker_id)
    if selected_name and selected_name != "Inconnu":
        return selected_name
    return speaker_id or "Inconnu"


def transcript_with_diarization(segments, speaker_names):
    """Build the portable, human-readable diarized transcript."""
    lines = []
    for segment in segments:
        speaker = speaker_label(segment.speaker_id, speaker_names)
        lines.append(f"[{format_timestamp(segment.start)}] {speaker}: {segment.text}")
    return "\n".join(lines)


def raw_transcript_as_text(segments):
    """Build a timestamped transcript before speaker attribution."""
    return "\n".join(
        f"[{format_timestamp(segment.start)}] {segment.text}" for segment in segments
    )


def transcript_as_markdown(segments, speaker_names):
    """Build the readable Markdown version of the final transcript."""
    lines = ["# Transcription", ""]
    for segment in segments:
        speaker = speaker_label(segment.speaker_id, speaker_names)
        lines.extend(
            [
                f"**[{format_timestamp(segment.start)}] {speaker}**",
                "",
                segment.text,
                "",
            ]
        )
    return "\n".join(lines)


def write_text_artifact(work, filename, content):
    """Persist an artifact so it remains available throughout the workflow."""
    path = Path(work) / filename
    path.write_text(content, encoding="utf-8")
    return path


def write_json_artifact(work, filename, content):
    """Persist a JSON artifact with readable Unicode text."""
    return write_text_artifact(
        work, filename, json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    )


def artifact_mime_type(path):
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".json": "application/json",
        ".md": "text/markdown",
        ".srt": "application/x-subrip",
    }.get(path.suffix, "text/plain")


def render_artifact_downloads(work):
    """Display one download button per generated artifact and a ZIP archive."""
    work = Path(work)
    artifact_names = [
        "audio.wav",
        "audio.mp3",
        "transcript_raw.txt",
        "transcript_raw.json",
        "diarization.json",
        "transcript.txt",
        "transcript.md",
        "transcript.json",
        "transcript-diarise.srt",
        "minutes.md",
    ]
    artifacts = [
        (work / name, name, name)
        for name in artifact_names
        if (work / name).is_file()
    ]
    source = st.session_state.get("source")
    if source and Path(source).is_file():
        source = Path(source)
        archive_name = source.name if source.name not in artifact_names else f"original_{source.name}"
        artifacts.insert(0, (source, archive_name, f"fichier source ({source.name})"))

    if not artifacts:
        return

    st.subheader("Téléchargements")
    columns = st.columns(2)
    for index, (path, _, label) in enumerate(artifacts):
        columns[index % len(columns)].download_button(
            f"Télécharger {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=artifact_mime_type(path),
            key=f"download_{work.name}_{index}",
        )

    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for path, archive_name, _ in artifacts:
            zip_file.writestr(archive_name, path.read_bytes())
    st.download_button(
        "Télécharger tous les fichiers (.zip)",
        data=archive.getvalue(),
        file_name="fichiers-reunion.zip",
        mime="application/zip",
        key=f"download_zip_{work.name}",
    )


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

tab_import, tab_transcription, tab_minutes, tab_export = st.tabs(
    ["Import", "Transcription", "Compte-rendu", "Export"]
)

with tab_import:
    st.subheader("1. Importer et préparer la réunion")
    uploaded_audio = st.file_uploader("Fichier audio", type=["mp3", "wav", "m4a"])
    video_audio = video_audio_uploader(key="local_video_to_audio")
    converted_audio = None

    if video_audio and not uploaded_audio:
        try:
            converted_audio = {
                "name": video_audio["name"],
                "data": base64.b64decode(video_audio["data"], validate=True),
            }
            st.success(f"✓ Audio extrait localement : {converted_audio['name']}")
        except (KeyError, ValueError) as error:
            st.error(f"Audio extrait par le navigateur invalide : {error}")
    selected_upload = uploaded_audio or converted_audio
    if selected_upload:
        st.caption("Fichier prêt : " + (selected_upload["name"] if isinstance(selected_upload, dict) else selected_upload.name))

    participants_text = st.text_area("Participants (un par ligne)")
    participants = participants_text.splitlines()
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
    expected = st.checkbox("Utiliser ce nombre comme nombre de locuteurs attendu", value=True)

    start_transcription = st.button(
        "Lancer la transcription",
        type="primary",
        disabled=selected_upload is None,
        help="Importez un fichier audio ou terminez la conversion vidéo pour activer ce bouton.",
    )

    if start_transcription and selected_upload:
        work = Path("work") / str(uuid.uuid4())
        work.mkdir(parents=True)
        source = work / Path(selected_upload["name"] if isinstance(selected_upload, dict) else selected_upload.name).name
        source.write_bytes(selected_upload["data"] if isinstance(selected_upload, dict) else selected_upload.getbuffer())
        st.session_state["work"] = work
        st.session_state["source"] = source
        st.session_state["participants"] = participants
        st.session_state["expected_speakers"] = expected
        st.session_state.pop("segments", None)
        st.session_state.pop("minutes", None)

        try:
            wav, mp3 = prepare_audio(source, work)
            st.success("✓ Audio préparé")
            Ollama(OLLAMA_BASE_URL).unload_all()
            raw = transcribe(wav, engine=asr_engine, language=language)
            write_text_artifact(work, "transcript_raw.txt", raw_transcript_as_text(raw))
            write_json_artifact(work, "transcript_raw.json", dump(raw))
            st.success("✓ Transcription terminée")
            dia = diarize(wav, len(participants) if expected else None)
            write_json_artifact(work, "diarization.json", dump(dia))
            st.success("✓ Diarisation terminée")
            aligned_segments = align_transcription_with_speakers(raw, dia)
            st.session_state["segments"] = merge_consecutive(aligned_segments)
            st.session_state["speaker_names"] = {}
            st.success("✓ Pipeline terminé — consultez l’onglet Transcription")
        except Exception as error:
            st.error(str(error))

with tab_transcription:
    st.subheader("2. Vérifier la transcription")
    if "segments" not in st.session_state:
        st.info("Importez un fichier et lancez le traitement depuis l’onglet Import.")
    else:
        segments = st.session_state["segments"]
        available_names = [p.strip() for p in st.session_state.get("participants", []) if p.strip()]
        mappings = st.session_state.setdefault("speaker_names", {})
        st.caption("Associez chaque identifiant de diarisation une seule fois. Ces choix restent modifiables.")
        mapping_columns = st.columns(min(3, max(1, len(speaker_ids(segments)))))
        for index, speaker_id in enumerate(speaker_ids(segments)):
            choices = available_names.copy()
            current_name = mappings.get(speaker_id) or speaker_id
            if current_name not in choices:
                choices.append(current_name)
            mappings[speaker_id] = mapping_columns[index % len(mapping_columns)].selectbox(
                speaker_id, choices, index=choices.index(current_name), key=f"speaker_mapping_{speaker_id}"
            )
        for segment in segments:
            segment.speaker_name = speaker_label(segment.speaker_id, mappings)
        transcript = transcript_with_diarization(segments, mappings)
        work = st.session_state["work"]
        write_text_artifact(work, "transcript.txt", transcript)
        write_text_artifact(work, "transcript.md", transcript_as_markdown(segments, mappings))
        write_json_artifact(work, "transcript.json", dump(segments))
        write_text_artifact(work, "transcript-diarise.srt", transcript_as_srt(segments, mappings))
        with st.expander("Transcript diarisé", expanded=True):
            st.caption("Les heures sont affichées au début de chaque prise de parole.")
            displayed_speakers = list(dict.fromkeys(speaker_label(s.speaker_id, mappings) for s in segments))
            last_speaker = object()
            dialogue_html = []
            for segment in segments:
                speaker = speaker_label(segment.speaker_id, mappings)
                background, text_color = speaker_color(speaker, displayed_speakers)
                header = ""
                if segment.speaker_id != last_speaker:
                    header = f'<div style="margin: 9px 0 1px; font-size: 0.85rem; font-weight: 600; color: {text_color};">{escape(speaker)} · {format_timestamp(segment.start)}</div>'
                    last_speaker = segment.speaker_id
                dialogue_html.append(f'{header}<div style="margin: 0 0 2px; padding: 5px 9px; border-radius: 6px; line-height: 1.35; background-color: {background}; color: {text_color};">{escape(segment.text)}</div>')
            st.markdown("".join(dialogue_html), unsafe_allow_html=True)

with tab_minutes:
    st.subheader("3. Générer le compte-rendu")
    if "segments" not in st.session_state:
        st.info("Terminez le traitement et la vérification dans les onglets précédents.")
    else:
        work = st.session_state["work"]
        transcript = Path(work / "transcript.txt").read_text(encoding="utf-8") if (work / "transcript.txt").is_file() else transcript_with_diarization(st.session_state["segments"], st.session_state.get("speaker_names", {}))
        prompt = st.text_area("Prompt du compte-rendu", Path("prompts/default_minutes.txt").read_text())
        if MINUTES_MODEL in models:
            st.caption(f"Modèle du compte-rendu : `{MINUTES_MODEL}` — reasoning désactivé.")
            if st.button("Générer le compte-rendu", type="primary"):
                response = Ollama(OLLAMA_BASE_URL).generate(MINUTES_MODEL, f"{prompt}\n\nVoici la transcription diarizée :\n{transcript}", think=False)
                write_text_artifact(work, "minutes.md", response)
                st.session_state["minutes"] = response
        elif models:
            st.error(f"Le modèle requis `{MINUTES_MODEL}` n'est pas installé dans Ollama. Installez-le ou définissez OLLAMA_MINUTES_MODEL avec son tag local exact.")
        if "minutes" in st.session_state:
            st.markdown(st.session_state["minutes"])

with tab_export:
    st.subheader("4. Exporter les résultats")
    if "work" not in st.session_state:
        st.info("Les fichiers apparaîtront ici après l’import et le traitement d’une réunion.")
    else:
        render_artifact_downloads(st.session_state["work"])
        st.caption("Les sous-titres .srt sont compatibles VLC : Sous-titres > Ajouter un fichier de sous-titres.")
