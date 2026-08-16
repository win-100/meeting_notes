from dataclasses import dataclass, field, asdict
@dataclass
class Word: start: float; end: float; text: str
@dataclass
class TranscriptSegment: start: float; end: float; text: str; speaker_id: str|None=None; speaker_name: str|None=None; words:list[Word]=field(default_factory=list)
@dataclass
class DiarizationSegment: start: float; end: float; speaker_id: str
def dump(items): return [asdict(x) for x in items]
