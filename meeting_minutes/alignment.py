from .models import TranscriptSegment


def align_transcription_with_speakers(transcript, diarization):
    """Assign each transcription segment to the speaker with most overlap.

    Ties are resolved deterministically by speaker identifier.
    """
    aligned_segments = []

    for transcript_segment in transcript:
        overlap_by_speaker = {}

        for diarization_segment in diarization:
            overlap = max(
                0,
                min(transcript_segment.end, diarization_segment.end)
                - max(transcript_segment.start, diarization_segment.start),
            )
            speaker_id = diarization_segment.speaker_id
            overlap_by_speaker[speaker_id] = (
                overlap_by_speaker.get(speaker_id, 0) + overlap
            )

        speaker_id = (
            max(overlap_by_speaker, key=lambda item: (overlap_by_speaker[item], item))
            if overlap_by_speaker
            else None
        )
        aligned_segments.append(
            TranscriptSegment(
                transcript_segment.start,
                transcript_segment.end,
                transcript_segment.text,
                speaker_id,
                words=transcript_segment.words,
            )
        )

    return aligned_segments


def merge_consecutive(segments, gap=1):
    """Merge adjacent segments from the same speaker within ``gap`` seconds."""
    merged_segments = []

    for segment in segments:
        previous_segment = merged_segments[-1] if merged_segments else None
        can_merge = (
            previous_segment
            and previous_segment.speaker_id == segment.speaker_id
            and segment.start - previous_segment.end <= gap
        )

        if can_merge:
            previous_segment.end = segment.end
            previous_segment.text += f" {segment.text}"
        else:
            merged_segments.append(segment)

    return merged_segments
