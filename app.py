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
from meeting_minutes.prompt_templates import (
    PROMPT_VARIABLES,
    format_special_terms,
    load_saved_templates,
    prompt_variables,
    render_prompt,
    save_templates,
)
from meeting_minutes.subtitles import transcript_as_srt


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MINUTES_MODEL = os.getenv("OLLAMA_MINUTES_MODEL", "qwen3.5:4b-q4_K_M")
PROMPTS_DIRECTORY = Path(__file__).resolve().parent / "prompts"
DEFAULT_MINUTES_PROMPT_PATH = PROMPTS_DIRECTORY / "default_minutes.txt"
DEFAULT_CONTEXT_PROMPT_PATH = PROMPTS_DIRECTORY / "default_context.txt"
SAVED_PROMPTS_PATH = PROMPTS_DIRECTORY / "saved_prompts.json"
DEFAULT_TEMPLATE_NAME = "Compte-rendu par défaut"

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
    """Display the MP3 separately and archive only the export documents."""
    work = Path(work)
    document_names = [
        "transcript.txt",
        "transcript.json",
        "transcript-diarise.srt",
        "minutes.md",
    ]
    documents = [
        (work / name, name, name)
        for name in document_names
        if (work / name).is_file()
    ]
    audio = work / "audio.mp3"

    if audio.is_file():
        st.subheader("Audio")
        st.download_button(
            "Télécharger l'audio MP3",
            data=audio.read_bytes(),
            file_name=audio.name,
            mime=artifact_mime_type(audio),
            key=f"download_audio_{work.name}",
        )

    if not documents:
        return

    st.subheader("Documents")
    columns = st.columns(2)
    for index, (path, _, label) in enumerate(documents):
        columns[index % len(columns)].download_button(
            f"Télécharger {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=artifact_mime_type(path),
            key=f"download_{work.name}_{index}",
        )

    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for path, archive_name, _ in documents:
            zip_file.writestr(archive_name, path.read_bytes())
    st.download_button(
        "Télécharger les documents (.zip)",
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


def default_minutes_prompt():
    """Read the shipped prompt so it can be copied into user templates."""
    return DEFAULT_MINUTES_PROMPT_PATH.read_text(encoding="utf-8")


def default_context_prompt():
    """Read the prompt used to infer a concise meeting context."""
    return DEFAULT_CONTEXT_PROMPT_PATH.read_text(encoding="utf-8")


def load_selected_minutes_template():
    """Load the selected template into the editable fields."""
    selected = st.session_state["minutes_template_selector"]
    saved_templates = load_saved_templates(SAVED_PROMPTS_PATH)
    st.session_state["minutes_prompt_editor"] = (
        default_minutes_prompt()
        if selected == DEFAULT_TEMPLATE_NAME
        else saved_templates.get(selected, default_minutes_prompt())
    )
    st.session_state["minutes_prompt_name"] = (
        "" if selected == DEFAULT_TEMPLATE_NAME else selected
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

tab_import, tab_transcription, tab_minutes, tab_export = st.tabs(
    ["Import", "Transcription", "Compte-rendu", "Export"]
)

with tab_import:
    st.subheader("1. Importer et préparer la réunion")
    uploaded_audio = st.file_uploader("Fichier audio", type=["mp3", "wav", "m4a"])
    video_audio = video_audio_uploader(key="local_video_to_audio")

    # A component value is sent during the rerun triggered by the browser
    # conversion.  The button itself triggers a later rerun, where that value
    # is not guaranteed to be returned again.  Keep a server-side snapshot so
    # clicking the button cannot make the converted video disappear.
    if video_audio:
        try:
            converted_audio = {
                "name": video_audio["name"],
                "data": base64.b64decode(video_audio["data"], validate=True),
            }
            st.session_state["converted_audio"] = converted_audio
            st.success(f"✓ Audio extrait localement : {converted_audio['name']}")
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Audio extrait par le navigateur invalide : {error}")
    converted_audio = st.session_state.get("converted_audio")
    # An audio file selected with Streamlit takes precedence.  Otherwise use
    # the snapshot above rather than the component's transient return value.
    selected_upload = uploaded_audio if uploaded_audio is not None else converted_audio
    if selected_upload:
        filename = selected_upload["name"] if isinstance(selected_upload, dict) else selected_upload.name
        st.caption(f"Fichier prêt : {filename}")

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
        filename = selected_upload["name"] if isinstance(selected_upload, dict) else selected_upload.name
        source = work / Path(filename).name
        source.write_bytes(selected_upload["data"] if isinstance(selected_upload, dict) else selected_upload.getbuffer())
        st.session_state["work"] = work
        st.session_state["source"] = source
        st.session_state["participants"] = participants
        st.session_state["expected_speakers"] = expected
        st.session_state.pop("segments", None)
        st.session_state.pop("minutes", None)
        st.session_state.pop("minutes_context", None)

        try:
            wav, mp3 = prepare_audio(source, work)
            st.success("✓ Audio préparé")
            Ollama(OLLAMA_BASE_URL).unload_all()
            asr_progress = st.progress(0, text="Préparation de la transcription…")

            def update_asr_progress(stage, completed, total):
                if stage == "Découpage de l'audio":
                    asr_progress.progress(5, text="Découpage de l'audio en segments…")
                elif completed == 0:
                    asr_progress.progress(10, text=f"{stage}…")
                else:
                    percent = 10 + round(75 * completed / total)
                    asr_progress.progress(
                        percent,
                        text=f"{stage} : segment {completed}/{total}",
                    )

            raw = transcribe(
                wav,
                engine=asr_engine,
                language=language,
                progress_callback=update_asr_progress,
            )
            asr_progress.progress(85, text="Enregistrement de la transcription…")
            write_text_artifact(work, "transcript_raw.txt", raw_transcript_as_text(raw))
            write_json_artifact(work, "transcript_raw.json", dump(raw))
            st.success("✓ Transcription terminée")
            asr_progress.progress(90, text="Diarisation des locuteurs…")
            dia = diarize(wav, len(participants) if expected else None)
            write_json_artifact(work, "diarization.json", dump(dia))
            st.success("✓ Diarisation terminée")
            aligned_segments = align_transcription_with_speakers(raw, dia)
            st.session_state["segments"] = merge_consecutive(aligned_segments)
            st.session_state["speaker_names"] = {}
            asr_progress.progress(100, text="Traitement terminé")
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
        st.caption(
            "Variables du prompt : `{contexte}`, `{mots_particuliers}` et "
            "`{transcript}`. Elles sont remplacées à la génération."
        )

        special_terms = st.text_area(
            "Mots particuliers à corriger (un par ligne)",
            value="OneStock\nChausséa",
            key="minutes_special_terms",
            help="Indiquez ici les orthographes de référence, par exemple des noms propres ou produits.",
        )
        formatted_terms = format_special_terms(special_terms)

        if models:
            default_model_index = models.index(MINUTES_MODEL) if MINUTES_MODEL in models else 0
            llm_model = st.selectbox(
                "Modèle Ollama",
                models,
                index=default_model_index,
                key="minutes_model",
            )
            st.caption("Le reasoning est désactivé pour limiter la durée de génération.")
            if st.button("Déduire le contexte depuis la conversation"):
                context_prompt = render_prompt(
                    default_context_prompt(),
                    {
                        "mots_particuliers": formatted_terms,
                        "transcript": transcript,
                    },
                )
                try:
                    with st.spinner("Analyse de la conversation par Ollama…"):
                        st.session_state["minutes_context"] = Ollama(
                            OLLAMA_BASE_URL
                        ).generate(llm_model, context_prompt, think=False).strip()
                    st.rerun()
                except Exception as error:
                    st.error(f"Impossible de déduire le contexte : {error}")
        else:
            llm_model = None
            st.error("Aucun modèle Ollama n'est disponible.")

        context = st.text_area(
            "Contexte complémentaire",
            key="minutes_context",
            placeholder="Ajoutez un contexte, ou utilisez le bouton ci-dessus pour le déduire de la conversation.",
            help="Ce texte est injecté à la place de {contexte} dans le prompt.",
        )

        saved_templates = load_saved_templates(SAVED_PROMPTS_PATH)
        template_options = [DEFAULT_TEMPLATE_NAME, *sorted(saved_templates)]
        pending_template = st.session_state.pop("minutes_template_pending", None)
        if pending_template in template_options:
            st.session_state["minutes_template_selector"] = pending_template
            load_selected_minutes_template()
        if "minutes_template_selector" not in st.session_state or st.session_state["minutes_template_selector"] not in template_options:
            st.session_state["minutes_template_selector"] = DEFAULT_TEMPLATE_NAME
        if "minutes_prompt_editor" not in st.session_state:
            st.session_state["minutes_prompt_editor"] = default_minutes_prompt()
        if "minutes_prompt_name" not in st.session_state:
            st.session_state["minutes_prompt_name"] = ""

        st.selectbox(
            "Prompt enregistré",
            template_options,
            key="minutes_template_selector",
            on_change=load_selected_minutes_template,
        )
        prompt = st.text_area(
            "Bloc prompt",
            key="minutes_prompt_editor",
            height=360,
            help="Éditez le gabarit ; les paramètres entre accolades seront injectés à la génération.",
        )
        template_name = st.text_input(
            "Nom pour enregistrer ce prompt",
            key="minutes_prompt_name",
            placeholder="Ex. Compte-rendu de comité projet",
        )
        save_column, delete_column = st.columns(2)
        if save_column.button("Enregistrer le prompt"):
            normalized_name = template_name.strip()
            if not normalized_name:
                st.error("Donnez un nom au prompt avant de l'enregistrer.")
            elif normalized_name == DEFAULT_TEMPLATE_NAME:
                st.error("Choisissez un autre nom : le prompt par défaut ne peut pas être remplacé.")
            else:
                saved_templates[normalized_name] = prompt
                save_templates(SAVED_PROMPTS_PATH, saved_templates)
                st.session_state["minutes_template_pending"] = normalized_name
                st.rerun()
        if delete_column.button(
            "Supprimer ce prompt",
            disabled=st.session_state["minutes_template_selector"] == DEFAULT_TEMPLATE_NAME,
        ):
            saved_templates.pop(st.session_state["minutes_template_selector"], None)
            save_templates(SAVED_PROMPTS_PATH, saved_templates)
            st.session_state["minutes_template_pending"] = DEFAULT_TEMPLATE_NAME
            st.rerun()

        variables = prompt_variables(prompt)
        unknown_variables = [variable for variable in variables if variable not in PROMPT_VARIABLES]
        if unknown_variables:
            st.warning(
                "Variables non reconnues conservées telles quelles : "
                + ", ".join(f"`{{{variable}}}`" for variable in unknown_variables)
            )
        missing_variables = [variable for variable in PROMPT_VARIABLES if variable not in variables]
        if missing_variables:
            st.info(
                "Variables non utilisées par ce prompt : "
                + ", ".join(f"`{{{variable}}}`" for variable in missing_variables)
            )

        if llm_model and st.button("Générer le compte-rendu", type="primary"):
            assembled_prompt = render_prompt(
                prompt,
                {
                    "contexte": context.strip() or "Aucun contexte complémentaire fourni.",
                    "mots_particuliers": formatted_terms,
                    "transcript": transcript,
                },
            )
            try:
                with st.spinner("Génération du compte-rendu par Ollama…"):
                    response = Ollama(OLLAMA_BASE_URL).generate(
                        llm_model, assembled_prompt, think=False
                    )
                write_text_artifact(work, "minutes.md", response)
                st.session_state["minutes"] = response
            except Exception as error:
                st.error(f"Impossible de générer le compte-rendu : {error}")
        if "minutes" in st.session_state:
            st.markdown(st.session_state["minutes"])

with tab_export:
    st.subheader("4. Exporter les résultats")
    if "work" not in st.session_state:
        st.info("Les fichiers apparaîtront ici après l’import et le traitement d’une réunion.")
    else:
        render_artifact_downloads(st.session_state["work"])
        st.caption("Les sous-titres .srt sont compatibles VLC : Sous-titres > Ajouter un fichier de sous-titres.")
