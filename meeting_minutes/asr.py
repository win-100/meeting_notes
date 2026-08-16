import gc, subprocess, torch
from pathlib import Path
from .models import TranscriptSegment
CHUNK_SECONDS=20
def transcribe(path):
 from nemo.collections.asr.models import ASRModel
 from omegaconf import open_dict
 device='cuda' if torch.cuda.is_available() else 'cpu'
 # Parakeet TDT/NeMo 3 mélange des activations FP32 dans son décodeur:
 # model.half() échoue (Float vs Half). La mémoire est donc libérée avant ce
 # transfert, notamment tout modèle Ollama laissé résident.
 model=None
 try:
  # Le try englobe le chargement : une erreur de transfert ne doit jamais
  # laisser les poids dans le processus Streamlit.
  torch.cuda.empty_cache() if device=='cuda' else None
  model=ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3').eval()
  # CUDA Graphs du décodeur TDT provoquent un accès mémoire illégal sur la
  # GTX 1660 SUPER au second segment. Le chemin non-graph est stable.
  with open_dict(model.cfg.decoding.greedy):
   model.cfg.decoding.greedy.use_cuda_graph_decoder=False
  model.change_decoding_strategy(model.cfg.decoding, verbose=False)
  model=model.to(device)
  # NeMo traite le fichier entier en une passe : sur 6 Go les activations
  # d'une réunion longue provoquent un OOM. Fenêtres fixes, une à la fois.
  path=Path(path); folder=path.parent/'asr_chunks'; folder.mkdir(exist_ok=True)
  subprocess.run(['ffmpeg','-y','-i',str(path),'-f','segment','-segment_time',str(CHUNK_SECONDS),'-c:a','pcm_s16le',str(folder/'chunk_%05d.wav')],check=True,capture_output=True)
  output=[]
  for index,chunk in enumerate(sorted(folder.glob('chunk_*.wav'))):
   with torch.inference_mode(): r=model.transcribe([str(chunk)],timestamps=True)[0]
   offset=index*CHUNK_SECONDS; ts=getattr(r,'timestamp',{}).get('segment',[]); text=getattr(r,'text','').strip()
   output += [TranscriptSegment(offset+float(x.get('start',0)),offset+float(x.get('end',0)),x.get('segment') or x.get('text','')) for x in ts if isinstance(x,dict)] or ([TranscriptSegment(offset,offset+CHUNK_SECONDS,text)] if text else [])
   if device=='cuda': torch.cuda.empty_cache()
  return output
 finally:
  del model
  gc.collect()
  if torch.cuda.is_available(): torch.cuda.empty_cache()
