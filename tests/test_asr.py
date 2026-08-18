import pytest

from meeting_minutes import asr
from meeting_minutes.asr import _chunk_boundaries


def test_chunk_boundaries_prefer_the_last_nearby_silence():
    assert _chunk_boundaries(46, [5, 16.4, 19.2, 35.5]) == [
        (0.0, 35.5),
        (35.5, 46),
    ]


def test_chunk_boundaries_keep_the_maximum_when_no_silence_is_nearby():
    assert _chunk_boundaries(43, [8, 12, 27]) == [
        (0.0, 40.0),
        (40.0, 43),
    ]


def test_transcribe_dispatches_to_whisper_with_the_selected_language(monkeypatch):
    expected = [object()]
    captured = {}
    monkeypatch.setattr(asr, "_split_on_silence", lambda path, directory: [(0, 40)])
    monkeypatch.setattr(
        asr,
        "_transcribe_whisper_turbo",
        lambda chunks, language: captured.update(chunks=chunks, language=language) or expected,
    )

    assert asr.transcribe("audio.wav", engine="whisper_turbo", language="fr") == expected
    assert captured["language"] == "fr"
    assert captured["chunks"][0][1:] == (0, 40)


def test_transcribe_dispatches_to_parakeet_without_language(monkeypatch):
    expected = [object()]
    monkeypatch.setattr(asr, "_split_on_silence", lambda path, directory: [(0, 40)])
    monkeypatch.setattr(asr, "_transcribe_parakeet", lambda chunks: expected)

    assert asr.transcribe("audio.wav", engine="parakeet", language="fr") == expected


def test_transcribe_rejects_unknown_engine():
    with pytest.raises(ValueError, match="inconnu"):
        asr.transcribe("audio.wav", engine="invalide")


def test_whisper_ignores_silence_and_does_not_condition_windows(monkeypatch):
    captured = {}

    class FakeSegment:
        start = 0
        end = 1
        text = " Bonjour "
        words = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([FakeSegment()]), None

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)

    result = asr._transcribe_whisper_turbo([("audio.wav", 40, 41)], "fr")

    assert result[0].text == "Bonjour"
    assert result[0].start == 40
    assert result[0].end == 41
    assert captured["vad_filter"] is True
    assert captured["vad_parameters"] == {"min_silence_duration_ms": 1000}
    assert captured["condition_on_previous_text"] is False
