import subprocess
from pathlib import Path
def prepare_audio(source,work):
 w=Path(work)/'audio.wav'; m=Path(work)/'audio.mp3'; b=['ffmpeg','-y','-i',str(source),'-vn','-ac','1','-ar','16000']; subprocess.run(b+['-c:a','pcm_s16le',str(w)],check=True,capture_output=True); subprocess.run(['ffmpeg','-y','-i',str(w),str(m)],check=True,capture_output=True); return w,m
