# Meeting Minutes

Application Streamlit locale pour préparer l'audio d'une réunion, la
transcrire, identifier les locuteurs, puis générer un compte-rendu avec Ollama.

Le traitement de l'audio, de la transcription et de la diarisation reste local.
Un token Hugging Face est uniquement requis pour télécharger et utiliser le
modèle de diarisation `pyannote/speaker-diarization-community-1`.

## Prérequis

- Python 3.10 ou plus récent ;
- [FFmpeg](https://ffmpeg.org/) installé et disponible dans le `PATH` ;
- Ollama lancé localement (par défaut sur `http://localhost:11434`) ;
- un modèle Ollama installé, par exemple `ollama pull qwen2.5:3b` ;
- un compte Hugging Face avec accès accepté au modèle de diarisation pyannote ;
- facultatif mais recommandé : un GPU NVIDIA compatible CUDA.

## Installation

Créez un environnement virtuel puis installez les dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Les dépendances PyTorch déclarées dans `requirements.txt` visent CUDA 12.8.
> Adaptez-les si votre installation CUDA est différente ou si vous utilisez le
> CPU.

Créez votre fichier de configuration depuis l'exemple :

```bash
cp .env.example .env
```

Renseignez au minimum le token Hugging Face :

```dotenv
HF_TOKEN=hf_votre_token
OLLAMA_BASE_URL=http://localhost:11434
```

Le fichier `.env` est ignoré par Git : ne le versionnez pas.

## Lancer l'application

Assurez-vous qu'Ollama est en cours d'exécution, puis lancez :

```bash
streamlit run app.py
```

Dans l'interface :

1. importez un fichier `.mp3`, `.wav`, `.m4a`, `.mp4` ou `.mkv` ;
2. indiquez les participants si vous les connaissez ;
3. lancez la préparation, la transcription et la diarisation ;
4. associez les identifiants de locuteurs aux participants ;
5. choisissez un modèle Ollama et générez le compte-rendu.

## Fonctionnement

Le pipeline produit un WAV mono 16 kHz et un MP3 dans `work/<uuid>/`, puis :

1. transcrit l'audio avec `nvidia/parakeet-tdt-0.6b-v3` ;
2. découpe l'audio en segments de 20 secondes pour limiter la mémoire vidéo ;
3. réalise la diarisation avec pyannote ;
4. aligne les segments de transcription avec le locuteur ayant le plus grand
   recouvrement temporel ;
5. envoie la transcription annotée au modèle Ollama sélectionné.

Les modèles sont libérés entre les étapes lourdes afin de mieux fonctionner sur
des cartes disposant de peu de VRAM.

## Structure du projet

```text
app.py                    Interface Streamlit
meeting_minutes/
  audio.py                Normalisation audio avec FFmpeg
  asr.py                  Transcription Parakeet/NeMo
  diarization.py          Identification des locuteurs avec pyannote
  alignment.py            Alignement transcription/locuteurs
  ollama_client.py        Client HTTP Ollama
  models.py               Modèles de données
prompts/default_minutes.txt  Prompt de compte-rendu par défaut
```

## Dépannage

- **Ollama inaccessible** : vérifiez que le service est lancé et que
  `OLLAMA_BASE_URL` est correct.
- **`HF_TOKEN manquant` ou accès refusé** : renseignez le token dans `.env` et
  acceptez les conditions du modèle pyannote sur Hugging Face.
- **`ffmpeg: command not found`** : installez FFmpeg avec le gestionnaire de
  paquets de votre système.
- **Mémoire GPU insuffisante** : fermez les applications utilisant le GPU ; le
  programme basculera sur CPU si CUDA n'est pas disponible, avec des temps de
  traitement plus longs.
