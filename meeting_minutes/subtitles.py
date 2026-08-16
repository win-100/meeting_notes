"""Helpers to export timestamped transcripts as SubRip subtitles."""


def format_srt_timestamp(seconds):
    """Format elapsed seconds as an SRT timestamp (``HH:MM:SS,mmm``)."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def transcript_as_srt(segments, speaker_names=None):
    """Return diarized transcript segments in the VLC-compatible SRT format."""
    speaker_names = speaker_names or {}
    entries = []

    for segment in segments:
        speaker = speaker_names.get(segment.speaker_id) or segment.speaker_id or "Inconnu"
        text = " ".join(segment.text.split())
        if not text:
            continue
        entries.append(
            "\n".join(
                [
                    str(len(entries) + 1),
                    f"{format_srt_timestamp(segment.start)} --> "
                    f"{format_srt_timestamp(max(segment.start, segment.end))}",
                    f"{speaker}: {text}",
                ]
            )
        )

    return "\n\n".join(entries) + ("\n" if entries else "")
