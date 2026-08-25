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
- le modèle Ollama `qwen3.5:4b-q4_K_M` installé ;
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
# Facultatif : adaptez ce tag s'il diffère dans votre installation Ollama.
OLLAMA_MINUTES_MODEL=qwen3.5:4b-q4_K_M
```

Le fichier `.env` est ignoré par Git : ne le versionnez pas.

## Lancer l'application

Assurez-vous qu'Ollama est en cours d'exécution, puis lancez :

```bash
streamlit run app.py
```

L'interface est organisée en quatre onglets qui suivent le workflow :

1. **Import** : importez un fichier audio (`.mp3`, `.wav`, `.m4a`) ou choisissez
   une vidéo (`.mp4`, `.mkv`) dans le convertisseur navigateur, puis renseignez
   les participants et lancez la préparation, la transcription et la diarisation ;
2. **Transcription** : associez les identifiants de locuteurs aux participants
   et vérifiez la transcription diarizée ;
3. **Compte-rendu** : choisissez un modèle Ollama, complétez ou déduisez le
   contexte depuis la conversation, gérez des prompts réutilisables et générez
   le compte-rendu ; le reasoning est désactivé ;
4. **Export** : téléchargez l'audio MP3 séparément, les documents produits ou
   leur archive `.zip`.

Pour afficher le fichier téléchargé dans VLC, ouvrez la vidéo puis sélectionnez
**Sous-titres > Ajouter un fichier de sous-titres** et choisissez le `.srt`.

### Vidéos : conversion dans le navigateur

Une vidéo est convertie en MP3 mono 16 kHz directement dans le navigateur ;
le fichier vidéo original ne traverse jamais la connexion Streamlit. Seul le
MP3 compressé est transmis à l'application, puis normalisé par FFmpeg avant la
transcription. Le convertisseur WebAssembly est chargé depuis jsDelivr au
premier usage. Pour les enregistrements longs, la taille maximale des messages
Streamlit est configurée dans `.streamlit/config.toml`.

Cette limite est configurée dans `.streamlit/config.toml` à `500` Mo pour
prendre en charge les enregistrements longs. Redémarrez Streamlit après toute
modification de cette valeur.

## Fonctionnement

Le pipeline produit un WAV mono 16 kHz et un MP3 dans `work/<uuid>/`, puis :

1. transcrit l'audio avec le moteur sélectionné (Whisper Turbo ou
   `nvidia/parakeet-tdt-0.6b-v3`) ;
2. découpe l'audio en segments d'au plus 20 secondes pour limiter la mémoire
   vidéo, en privilégiant un silence proche de la limite ;
3. réalise la diarisation avec pyannote ;
4. aligne les segments de transcription avec le locuteur ayant le plus grand
   recouvrement temporel ;
5. envoie la transcription annotée au modèle Ollama choisi via Ollama, avec le
   reasoning désactivé.

### Prompts de compte-rendu

L'onglet **Compte-rendu** utilise un bloc de prompt éditable. Il peut contenir
les variables suivantes, remplacées au moment de la génération :

- `{contexte}` : le contexte saisi manuellement ou déduit avec Ollama ;
- `{mots_particuliers}` : les orthographes de référence, une par ligne ;
- `{transcript}` : la transcription diarizée de la réunion.

Le prompt fourni par défaut et le prompt de déduction du contexte se trouvent
respectivement dans `prompts/default_minutes.txt` et
`prompts/default_context.txt`. Les prompts enregistrés depuis l'interface sont
conservés localement dans `prompts/saved_prompts.json` (ignoré par Git).

Les modèles sont libérés entre les étapes lourdes afin de mieux fonctionner sur
des cartes disposant de peu de VRAM.

### Fichiers téléchargeables

La section **Audio** propose uniquement `audio.mp3`, dans un téléchargement
séparé. La section **Documents** propose la transcription finale (`.txt` et
`.json`), les sous-titres `.srt` et, après génération, `minutes.md`. Son ZIP
contient seulement ces documents : il n'inclut pas l'audio.

### Moteur et langue de transcription

Deux moteurs locaux sont disponibles :

- **Whisper Turbo** (par défaut) : choisissez explicitement la langue de la
  réunion. Pour une réunion française, conservez **Français** ; cela évite une
  détection erronée en anglais.
- **Parakeet TDT v3** : conserve son comportement historique et détecte
  automatiquement la langue.

Whisper Turbo est téléchargé automatiquement au premier lancement. Il utilise
`faster-whisper` en INT8, afin de rester compatible avec les GPU à 6 Go de VRAM.
La détection de parole est activée : elle évite notamment que Whisper interprète
un silence final comme une mention de sous-titrage.

## Structure du projet

```text
app.py                    Interface Streamlit
meeting_minutes/
  audio.py                Normalisation audio avec FFmpeg
  asr.py                  Transcription Parakeet/NeMo
  diarization.py          Identification des locuteurs avec pyannote
  alignment.py            Alignement transcription/locuteurs
  ollama_client.py        Client HTTP Ollama
  prompt_templates.py     Variables et stockage des prompts
  models.py               Modèles de données
prompts/default_minutes.txt  Prompt de compte-rendu par défaut
prompts/default_context.txt  Prompt de déduction du contexte
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
