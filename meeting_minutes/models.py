from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker_id: str


def dump(items):
    """Convert dataclass instances to dictionaries for serialization."""
    return [asdict(item) for item in items]
