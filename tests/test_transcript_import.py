import pytest

from meeting_minutes.transcript_import import load_diarized_transcript


def test_loads_application_json_export_and_keeps_speaker_name():
    segments = load_diarized_transcript(
        "transcript.json",
        '[{"start": 1.25, "end": 3, "text": " Bonjour ", '
        '"speaker_id": "SPEAKER_00", "speaker_name": "Alice"}]',
    )

    assert len(segments) == 1
    assert segments[0].start == 1.25
    assert segments[0].speaker_id == "SPEAKER_00"
    assert segments[0].speaker_name == "Alice"
    assert segments[0].text == "Bonjour"


def test_loads_timestamped_text_export_and_infers_end_times():
    segments = load_diarized_transcript(
        "transcript.txt",
        "[00:00:02] Alice: Bonjour\n[00:00:05] Bob: Salut\n",
    )

    assert [(segment.start, segment.end, segment.speaker_id) for segment in segments] == [
        (2.0, 5.0, "Alice"),
        (5.0, 6.0, "Bob"),
    ]


def test_loads_srt_export():
    segments = load_diarized_transcript(
        "transcript-diarise.srt",
        "1\n00:00:01,000 --> 00:00:02,500\nAlice: Bonjour\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nBob: Salut\n",
    )

    assert [(segment.start, segment.end, segment.speaker_name, segment.text) for segment in segments] == [
        (1.0, 2.5, "Alice", "Bonjour"),
        (3.0, 4.0, "Bob", "Salut"),
    ]


def test_loads_markdown_export():
    segments = load_diarized_transcript(
        "transcript.md",
        "# Transcription\n\n**[00:00:01] Alice**\n\nBonjour tout le monde.\n",
    )

    assert segments[0].speaker_id == "Alice"
    assert segments[0].text == "Bonjour tout le monde."


def test_rejects_non_diarized_text_export():
    with pytest.raises(ValueError, match="format"):
        load_diarized_transcript("transcript.txt", "[00:00:01] Bonjour")
