"""Read the diarized transcript formats exported by the application."""

import json
import re
from pathlib import Path

from .models import TranscriptSegment, Word


_TEXT_LINE = re.compile(
    r"^\s*\[(?P<timestamp>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)\]\s*"
    r"(?P<speaker>[^:]+):\s*(?P<text>.+?)\s*$"
)
_MARKDOWN_HEADER = re.compile(
    r"^\s*\*\*\[(?P<timestamp>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)\]\s*"
    r"(?P<speaker>.+?)\*\*\s*$"
)
_SRT_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)
_SPEAKER_TEXT = re.compile(r"^\s*(?P<speaker>[^:]+):\s*(?P<text>.+?)\s*$", re.DOTALL)


def _seconds(value):
    """Convert an HH:MM:SS timestamp to seconds."""
    value = str(value).strip().replace(",", ".")
    try:
        hours, minutes, seconds = value.split(":")
        result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Horodatage invalide : {value}") from error
    if result < 0:
        raise ValueError(f"Horodatage invalide : {value}")
    return result


def _normalize_segments(segments):
    """Validate segment data and infer a short end time when it is absent."""
    if not segments:
        raise ValueError("Le fichier ne contient aucun segment diarisé.")

    normalized = []
    for index, segment in enumerate(segments, start=1):
        text = " ".join(str(segment.text).split())
        speaker_id = (segment.speaker_id or "").strip()
        if not speaker_id:
            raise ValueError(f"Le segment {index} ne contient pas de locuteur.")
        if not text:
            raise ValueError(f"Le segment {index} ne contient pas de texte.")
        if segment.start < 0:
            raise ValueError(f"Le segment {index} a un début invalide.")
        normalized.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end) if segment.end is not None else None,
                text=text,
                speaker_id=speaker_id,
                speaker_name=segment.speaker_name or speaker_id,
                words=segment.words,
            )
        )

    for index, segment in enumerate(normalized):
        next_start = (
            normalized[index + 1].start if index + 1 < len(normalized) else None
        )
        if segment.end is None or segment.end < segment.start:
            segment.end = max(
                segment.start + 1,
                next_start if next_start is not None else segment.start + 1,
            )
    return normalized


def _parse_json(content):
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Le fichier JSON est invalide.") from error
    if isinstance(payload, dict):
        payload = payload.get("segments")
    if not isinstance(payload, list):
        raise ValueError("Le JSON doit contenir une liste de segments.")

    segments = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Le segment JSON {index} est invalide.")
        speaker_id = item.get("speaker_id") or item.get("speaker_name")
        words = [
            Word(float(word["start"]), float(word["end"]), str(word["text"]))
            for word in item.get("words", [])
            if isinstance(word, dict)
            and {"start", "end", "text"}.issubset(word)
        ]
        try:
            segments.append(
                TranscriptSegment(
                    start=float(item["start"]),
                    end=float(item["end"]) if item.get("end") is not None else None,
                    text=str(item.get("text", "")),
                    speaker_id=str(speaker_id) if speaker_id is not None else None,
                    speaker_name=item.get("speaker_name"),
                    words=words,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Le segment JSON {index} est invalide.") from error
    return _normalize_segments(segments)


def _parse_timestamped_text(content):
    segments = []
    for line in content.splitlines():
        match = _TEXT_LINE.match(line)
        if match:
            segments.append(
                TranscriptSegment(
                    start=_seconds(match["timestamp"]),
                    end=None,
                    text=match["text"],
                    speaker_id=match["speaker"].strip(),
                    speaker_name=match["speaker"].strip(),
                )
            )
    if not segments:
        raise ValueError(
            "Le TXT doit utiliser le format « [00:00:00] Locuteur: texte » exporté par l’application."
        )
    return _normalize_segments(segments)


def _parse_markdown(content):
    segments = []
    current = None
    for line in content.splitlines():
        match = _MARKDOWN_HEADER.match(line)
        if match:
            if current is not None:
                current.text = " ".join(current.text.split())
                segments.append(current)
            speaker = match["speaker"].strip()
            current = TranscriptSegment(
                start=_seconds(match["timestamp"]),
                end=None,
                text="",
                speaker_id=speaker,
                speaker_name=speaker,
            )
        elif current is not None and line.strip():
            current.text = f"{current.text} {line.strip()}"
    if current is not None:
        current.text = " ".join(current.text.split())
        segments.append(current)
    if not segments:
        raise ValueError("Le Markdown ne contient pas de transcript diarisé exporté par l’application.")
    return _normalize_segments(segments)


def _parse_srt(content):
    segments = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines and lines[0].isdigit():
            lines = lines[1:]
        if len(lines) < 2:
            continue
        timing = _SRT_TIME_RANGE.match(lines[0])
        speaker_text = _SPEAKER_TEXT.match("\n".join(lines[1:]))
        if not timing or not speaker_text:
            raise ValueError("Le SRT doit contenir « Locuteur: texte » dans chaque sous-titre.")
        speaker = speaker_text["speaker"].strip()
        segments.append(
            TranscriptSegment(
                start=_seconds(timing["start"]),
                end=_seconds(timing["end"]),
                text=speaker_text["text"],
                speaker_id=speaker,
                speaker_name=speaker,
            )
        )
    return _normalize_segments(segments)


def load_diarized_transcript(filename, content):
    """Parse an exported diarized transcript and return its segments.

    Supported extensions are JSON, TXT, SRT and the Markdown transcript export.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return _parse_json(content)
    if suffix == ".srt":
        return _parse_srt(content)
    if suffix == ".md":
        return _parse_markdown(content)
    if suffix == ".txt":
        return _parse_timestamped_text(content)
    raise ValueError("Format non pris en charge. Importez un fichier JSON, TXT, SRT ou Markdown.")
