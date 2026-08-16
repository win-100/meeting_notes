from meeting_minutes.asr import _chunk_boundaries


def test_chunk_boundaries_prefer_the_last_nearby_silence():
    assert _chunk_boundaries(46, [5, 16.4, 19.2, 35.5]) == [
        (0.0, 19.2),
        (19.2, 35.5),
        (35.5, 46),
    ]


def test_chunk_boundaries_keep_the_maximum_when_no_silence_is_nearby():
    assert _chunk_boundaries(43, [8, 12, 27]) == [
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 43),
    ]
