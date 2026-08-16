from .models import TranscriptSegment
def align_transcription_with_speakers(transcript, diarization):
 """Locuteur au recouvrement cumulé maximal; égalité déterministe par ID."""
 out=[]
 for s in transcript:
  o={}
  for d in diarization: o[d.speaker_id]=o.get(d.speaker_id,0)+max(0,min(s.end,d.end)-max(s.start,d.start))
  out.append(TranscriptSegment(s.start,s.end,s.text,max(o,key=lambda x:(o[x],x)) if o else None,words=s.words))
 return out
def merge_consecutive(xs,gap=1):
 out=[]
 for x in xs:
  if out and out[-1].speaker_id==x.speaker_id and x.start-out[-1].end<=gap: out[-1].end=x.end; out[-1].text+=' '+x.text
  else: out.append(x)
 return out
