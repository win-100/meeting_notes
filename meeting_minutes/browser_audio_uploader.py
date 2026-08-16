"""Streamlit component used to extract a video's audio in the browser."""

from pathlib import Path

import streamlit.components.v1 as components


_component = components.declare_component(
    "browser_audio_uploader",
    path=str(Path(__file__).parent / "components" / "browser_audio_uploader"),
)


def video_audio_uploader(*, key: str):
    """Return an audio payload produced locally by the browser.

    The returned dictionary contains a base64-encoded MP3, its display name and
    MIME type.  Crucially, the video never crosses the Streamlit connection.
    """
    return _component(key=key, default=None)
