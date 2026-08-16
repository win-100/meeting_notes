import requests
class Ollama:
 def __init__(self,url): self.url=url.rstrip('/')
 def models(self): return [x['name'] for x in requests.get(self.url+'/api/tags',timeout=3).json().get('models',[])]
 def generate(self,model,prompt):
  r=requests.post(self.url+'/api/generate',json={'model':model,'prompt':prompt,'stream':False},timeout=1800);r.raise_for_status();return r.json()['response']
 def unload_all(self):
  """Libère les modèles Ollama de la VRAM avant ASR/diarisation."""
  try:
   for item in requests.get(self.url+'/api/ps',timeout=3).json().get('models',[]):
    requests.post(self.url+'/api/generate',json={'model':item['name'],'keep_alive':0},timeout=10)
  except requests.RequestException: pass
