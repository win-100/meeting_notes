from meeting_minutes.models import TranscriptSegment
from meeting_minutes.subtitles import format_srt_timestamp, transcript_as_srt


def test_format_srt_timestamp_keeps_milliseconds_and_rolls_over():
    assert format_srt_timestamp(3661.9996) == "01:01:02,000"


def test_transcript_as_srt_includes_named_speakers_and_valid_cues():
    segments = [
        TranscriptSegment(0, 1.25, " Bonjour\nle monde ", "SPEAKER_00"),
        TranscriptSegment(1.5, 3, "À bientôt", "SPEAKER_01"),
    ]

    assert transcript_as_srt(segments, {"SPEAKER_00": "Alice"}) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "Alice: Bonjour le monde\n\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "SPEAKER_01: À bientôt\n"
    )
