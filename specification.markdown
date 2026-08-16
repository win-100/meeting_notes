# Application locale de transcription et compte-rendu de réunions

## 1. Objectif

Créer une application web locale en Python permettant de transformer un enregistrement de réunion en :

1. fichier audio normalisé ;
2. transcription avec timestamps ;
3. transcription diarizée indiquant qui parle ;
4. compte-rendu de réunion généré par un LLM local via Ollama ;
5. ensemble de fichiers intermédiaires téléchargeables.

L'application doit fonctionner principalement **en local**, sans dépendre d'une API cloud pour la transcription, la diarisation ou la génération du compte-rendu.

L'utilisateur dispose d'un PC Linux Ubuntu avec :

* GPU NVIDIA GTX 1660 SUPER ;
* 6 Go de VRAM ;
* pilotes NVIDIA/CUDA fonctionnels ;
* Ollama déjà installé ;
* modèles Ollama actuellement présents :

  * `qwen2.5-coder:1.5b`
  * `qwen2.5-coder:3b`

Un modèle généraliste pourra être ajouté ultérieurement, probablement `qwen2.5:3b`.

L'application doit être pensée pour des réunions d'une durée potentiellement importante, typiquement 30 minutes à plusieurs heures.

---

## 2. Stack souhaitée

### Interface

Utiliser :

```text
Streamlit
```

L'application doit pouvoir être lancée simplement avec :

```bash
streamlit run app.py
```

### Speech-to-text

Modèle par défaut :

```text
nvidia/parakeet-tdt-0.6b-v3
```

Objectifs :

* fonctionnement local ;
* utilisation du GPU NVIDIA ;
* français et anglais ;
* timestamps suffisamment précis pour permettre l'alignement avec la diarisation.

Ne pas utiliser l'API OpenAI Whisper dans la V1.

Le code doit isoler l'implémentation ASR derrière une interface claire afin de pouvoir remplacer ultérieurement Parakeet par un autre modèle.

Par exemple :

```python
transcribe(audio_path) -> TranscriptionResult
```

### Diarisation

Utiliser :

```text
pyannote/speaker-diarization-community-1
```

avec `pyannote.audio`.

Prévoir l'utilisation d'un token Hugging Face fourni dans :

```text
.env
```

par exemple :

```text
HF_TOKEN=...
```

Le programme ne doit jamais contenir ce token en dur.

Utiliser le GPU lorsque disponible.

Utiliser de préférence :

```python
output.exclusive_speaker_diarization
```

pour faciliter l'alignement avec les timestamps de transcription.

### Génération du compte-rendu

Utiliser Ollama localement.

URL par défaut :

```text
http://localhost:11434
```

L'application doit récupérer dynamiquement la liste des modèles installés auprès d'Ollama et proposer un menu déroulant.

Ne pas coder en dur `qwen2.5`.

Le modèle choisi par l'utilisateur sera utilisé pour produire le compte-rendu.

---

# 3. Contraintes matérielles

La machine ne dispose que de :

```text
6 Go de VRAM
```

Le programme doit donc être conçu pour éviter de garder plusieurs gros modèles simultanément en VRAM.

Pipeline souhaité :

```text
ASR
 ↓
libération des ressources ASR
 ↓
diarisation
 ↓
libération des ressources diarisation
 ↓
Ollama
 ↓
génération du compte-rendu
```

Faire au minimum :

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

lorsque cela est pertinent.

Ne pas utiliser BF16 sur cette machine.

Privilégier FP16 lorsque le modèle et le GPU le permettent.

Le programme doit également pouvoir retomber sur le CPU avec un message explicite si CUDA n'est pas disponible, même si les performances seront moins bonnes.

---

# 4. Formats d'entrée

L'utilisateur doit pouvoir uploader au minimum :

```text
.mp3
.wav
.m4a
.mp4
.mkv
```

Si le fichier contient de la vidéo, extraire l'audio.

Utiliser de préférence FFmpeg pour les conversions audio plutôt que de dépendre exclusivement de MoviePy.

Produire un fichier de travail standardisé, par exemple :

```text
meeting.wav
```

Format recommandé pour les traitements :

```text
mono
16 kHz
PCM WAV
```

Un MP3 pourra également être généré pour téléchargement par l'utilisateur.

---

# 5. Interface utilisateur

## 5.1 Chargement du fichier

En haut de page :

```text
Fichier de réunion
[ Choisir un fichier ]
```

Afficher ensuite :

* nom du fichier ;
* taille ;
* durée audio lorsque celle-ci est disponible.

---

## 5.2 Participants

Permettre à l'utilisateur d'indiquer les personnes censées participer à la réunion.

Interface souhaitée :

```text
Participants

[ Vincent                 ]
[ Charles                 ]
[ Sophie                  ]

[ + Ajouter un participant ]
```

Il doit être possible d'ajouter ou supprimer des participants.

### Nombre de speakers

Si l'utilisateur renseigne exactement N participants, proposer une option :

```text
☑ Utiliser ce nombre comme nombre de locuteurs attendu
```

Si cette option est cochée, transmettre à pyannote :

```python
num_speakers=N
```

Sinon laisser pyannote déterminer le nombre de speakers.

Prévoir également à terme la possibilité de définir :

```text
minimum de speakers
maximum de speakers
```

mais ce n'est pas indispensable dans la première UI.

---

# 6. Pipeline de traitement

Le pipeline doit être décomposé en étapes visibles.

```text
1. Préparation audio
2. Transcription
3. Diarisation
4. Alignement transcription / speakers
5. Identification des speakers
6. Génération du compte-rendu
```

Afficher une progression dans Streamlit.

Exemple :

```text
✓ Audio préparé
✓ Transcription terminée
✓ Diarisation terminée
● Identification des intervenants
○ Compte-rendu
```

Une erreur dans une étape ne doit pas supprimer les résultats déjà obtenus.

---

# 7. Transcription

La transcription doit produire une structure interne contenant au minimum :

```python
[
    {
        "start": 12.30,
        "end": 15.80,
        "text": "Bonjour, je propose qu'on commence..."
    },
    ...
]
```

Si Parakeet permet d'obtenir des timestamps mot par mot, les conserver également dans le JSON brut.

La transcription doit rester disponible indépendamment de la diarisation.

Créer :

```text
transcript_raw.json
transcript_raw.txt
```

---

# 8. Diarisation

Pyannote doit produire des segments comme :

```python
[
    {
        "start": 12.1,
        "end": 18.5,
        "speaker": "SPEAKER_00"
    },
    {
        "start": 18.5,
        "end": 27.3,
        "speaker": "SPEAKER_01"
    }
]
```

Conserver également ces données dans :

```text
diarization.json
```

---

# 9. Alignement transcription / diarisation

Créer une fonction dédiée :

```python
align_transcription_with_speakers(...)
```

Elle doit produire quelque chose comme :

```python
[
    {
        "start": 12.3,
        "end": 18.5,
        "speaker_id": "SPEAKER_00",
        "text": "Bonjour, je propose qu'on commence..."
    },
    {
        "start": 18.7,
        "end": 27.1,
        "speaker_id": "SPEAKER_01",
        "text": "Oui, j'ai justement regardé..."
    }
]
```

Utiliser autant que possible l'`exclusive_speaker_diarization` de Community-1.

Lorsque plusieurs speakers chevauchent une phrase ou un segment ASR, utiliser une stratégie déterministe et documentée, par exemple le speaker ayant la plus grande durée de recouvrement.

Éviter une logique basée uniquement sur le timestamp de début.

---

# 10. Identification nominative des speakers

C'est un point important de l'application.

La diarisation produira :

```text
SPEAKER_00
SPEAKER_01
SPEAKER_02
```

mais pas directement :

```text
Vincent
Charles
Sophie
```

La V1 doit donc fournir une étape de mapping.

## 10.1 Proposition automatique

Lorsque des participants ont été indiqués, utiliser le LLM Ollama pour essayer de déduire leur identité à partir du contenu de la conversation.

Exemple de prompt interne :

```text
Les participants connus de cette réunion sont :

- Vincent
- Charles

Voici une transcription dans laquelle les locuteurs sont identifiés
temporairement comme SPEAKER_00, SPEAKER_01, etc.

À partir des indices explicites présents dans la conversation
(prénoms employés, présentation, façon dont les personnes
s'interpellent), propose si possible une correspondance.

Ne devine pas lorsque les informations sont insuffisantes.

Retourne uniquement un JSON structuré.
```

Exemple :

```json
{
  "SPEAKER_00": {
    "name": "Vincent",
    "confidence": "high"
  },
  "SPEAKER_01": {
    "name": "Charles",
    "confidence": "medium"
  }
}
```

La suggestion du LLM ne doit jamais être appliquée silencieusement.

---

# 11. Validation manuelle des speakers

Après transcription et diarisation, arrêter le workflow avant la génération définitive du compte-rendu.

Afficher une interface du type :

```text
Intervenants détectés

SPEAKER_00

Extrait :
"Bonjour Charles. De mon côté j'ai repris le planning..."

Nom :
[ Vincent ▼ ]


SPEAKER_01

Extrait :
"Oui Vincent, j'ai vu ton mail hier..."

Nom :
[ Charles ▼ ]
```

Le menu doit contenir :

* tous les participants saisis ;
* `Inconnu`;
* éventuellement la possibilité de saisir un autre nom.

Afficher plusieurs extraits représentatifs de chaque speaker si possible.

Idéalement permettre d'écouter un court extrait audio correspondant au speaker directement dans Streamlit.

Le mapping final devient par exemple :

```python
{
    "SPEAKER_00": "Vincent",
    "SPEAKER_01": "Charles"
}
```

---

# 12. Transcript final

Après validation du mapping, générer un transcript Markdown propre.

Exemple :

```markdown
# Transcription

**[00:00:12] Vincent**

Bonjour Charles. De mon côté j'ai repris le planning et...

**[00:00:19] Charles**

Oui, j'ai vu ton mail hier. Il reste surtout...
```

Éviter de créer une nouvelle entrée lorsque le même speaker possède plusieurs segments consécutifs très courts.

Fusionner intelligemment les segments consécutifs du même speaker lorsque l'écart temporel est faible.

Produire :

```text
transcript.md
transcript.txt
transcript.json
```

Le JSON doit conserver les timestamps numériques et les IDs de speaker initiaux.

---

# 13. Prompt du compte-rendu

Afficher un grand champ texte éditable.

Exemple :

```text
Prompt du compte-rendu
┌───────────────────────────────────────┐
│ Voici la transcription d'une réunion │
│ ...                                   │
└───────────────────────────────────────┘
```

Précharger un prompt par défaut dans :

```text
prompts/default_minutes.txt
```

L'utilisateur peut le modifier avant chaque génération.

Ne pas enregistrer automatiquement ses modifications comme nouveau prompt par défaut.

Prévoir cependant une architecture permettant d'ajouter ultérieurement plusieurs templates de prompts.

---

# 14. Génération du compte-rendu avec Ollama

Interroger Ollama localement.

Récupérer les modèles installés via son API plutôt qu'une commande shell lorsque c'est possible.

Afficher :

```text
Modèle Ollama
[ qwen2.5:3b ▼ ]
```

Si Ollama n'est pas disponible :

```text
Ollama n'est pas accessible sur http://localhost:11434
```

et empêcher uniquement l'étape de génération du compte-rendu.

Les étapes transcription et diarisation doivent continuer à fonctionner.

Le prompt envoyé au LLM doit être :

```text
<USER_PROMPT>

Voici la transcription diarizée :

<TRANSCRIPT>
```

Le compte-rendu doit être rendu en Markdown.

Créer :

```text
minutes.md
```

---

# 15. Gestion des longues transcriptions

Ne pas supposer que tout transcript peut tenir dans le contexte du modèle Ollama.

Créer une stratégie pour les longues réunions.

Approche souhaitée :

```text
Transcript
 ↓
découpage en blocs suffisamment grands
 ↓
résumé structuré de chaque bloc
 ↓
fusion des résumés
 ↓
compte-rendu final
```

Le découpage doit si possible respecter les tours de parole plutôt que couper arbitrairement du texte.

Conserver autant que possible :

* décisions ;
* actions ;
* questions ouvertes ;
* désaccords ;
* informations importantes ;
* noms et responsabilités ;
* chiffres et dates.

Ne pas faire de résumé intermédiaire si le transcript tient directement dans le contexte disponible.

---

# 16. Fichiers téléchargeables

L'utilisateur doit pouvoir télécharger individuellement :

```text
original_filename_audio.mp3
transcript_raw.txt
transcript_raw.json
diarization.json
transcript.txt
transcript.md
transcript.json
minutes.md
```

Ajouter également :

```text
Télécharger tous les fichiers (.zip)
```

Le ZIP doit comprendre les fichiers disponibles, même si toutes les étapes n'ont pas été réalisées.

---

# 17. Gestion des fichiers temporaires

Créer pour chaque traitement un workspace temporaire distinct.

Exemple :

```text
work/
└── <job_uuid>/
    ├── original.mp4
    ├── audio.wav
    ├── audio.mp3
    ├── transcript_raw.json
    ├── diarization.json
    ├── transcript.json
    └── minutes.md
```

Éviter tout conflit si plusieurs fichiers portant le même nom sont utilisés.

Ne jamais utiliser des fichiers globaux tels que :

```text
segment_0.mp3
segment_1.mp3
```

dans le répertoire principal.

---

# 18. Architecture du projet

Structure cible :

```text
meeting-minutes/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── meeting_minutes/
│   ├── __init__.py
│   ├── audio.py
│   ├── asr.py
│   ├── diarization.py
│   ├── alignment.py
│   ├── speakers.py
│   ├── ollama_client.py
│   ├── minutes.py
│   ├── exporters.py
│   └── models.py
│
├── prompts/
│   └── default_minutes.txt
│
└── tests/
    ├── test_alignment.py
    ├── test_speaker_mapping.py
    ├── test_exporters.py
    └── test_ollama_client.py
```

Éviter un gros `app.py` monolithique.

L'interface Streamlit doit principalement orchestrer les fonctions présentes dans `meeting_minutes/`.

---

# 19. Modèles de données

Utiliser des dataclasses ou Pydantic.

Exemple conceptuel :

```python
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

@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker_id: str
```

Ne pas faire circuler uniquement des dictionnaires non typés dans toute l'application.

---

# 20. Gestion du GPU

Au démarrage afficher dans une zone "Système" :

```text
GPU : NVIDIA GeForce GTX 1660 SUPER
CUDA : disponible
VRAM : ~6 Go
Ollama : disponible
```

Ne pas considérer l'absence de `nvidia-smi` comme suffisante pour conclure que PyTorch voit CUDA.

Tester également :

```python
torch.cuda.is_available()
```

Afficher des messages utilisateur compréhensibles plutôt que les stack traces brutes.

Les stack traces peuvent être enregistrées dans les logs.

---

# 21. Cache Streamlit

Faire attention au cache des modèles.

Pour une machine à 6 Go de VRAM, ne pas utiliser naïvement :

```python
@st.cache_resource
```

pour maintenir simultanément Parakeet et pyannote en GPU.

Le choix de stratégie doit privilégier la stabilité mémoire.

Documenter précisément le cycle de vie des modèles.

---

# 22. Configuration

Créer :

```text
.env.example
```

avec :

```text
HF_TOKEN=
OLLAMA_BASE_URL=http://localhost:11434
```

Ne pas mettre de secret dans Git.

Ajouter à `.gitignore` :

```text
.env
work/
__pycache__/
.venv/
```

---

# 23. Installation

Le README doit expliquer précisément l'installation sous Ubuntu.

Au minimum :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Vérifier également la présence de :

```bash
ffmpeg
ollama
nvidia-smi
```

Fournir les commandes de vérification.

Expliquer la procédure Hugging Face nécessaire pour :

```text
pyannote/speaker-diarization-community-1
```

notamment :

1. créer un compte Hugging Face ;
2. accepter les conditions d'accès au modèle ;
3. créer un token ;
4. placer le token dans `.env`.

---

# 24. Détection d'Ollama

Au démarrage, appeler l'API Ollama locale pour récupérer la liste des modèles.

Ne pas faire :

```python
subprocess.run(["ollama", "list"])
```

sauf fallback nécessaire.

L'application doit être capable de trouver au minimum les modèles actuellement installés :

```text
qwen2.5-coder:1.5b
qwen2.5-coder:3b
```

mais elle ne doit pas les coder en dur.

Si `qwen2.5:3b` est ultérieurement installé, il doit apparaître automatiquement dans le menu sans modification du code.

---

# 25. Première version : identification vocale hors scope

Ne pas implémenter dans la V1 la reconnaissance biométrique persistante du type :

```text
empreinte vocale de Vincent
     ↓
reconnaissance automatique dans toutes les réunions
```

Conserver néanmoins l'architecture suffisamment modulaire pour pouvoir ajouter ultérieurement un module :

```text
speaker_recognition.py
```

La V1 utilise :

```text
diarisation
+
suggestion LLM
+
validation manuelle
```

---

# 26. Confidentialité

Le comportement par défaut doit être local.

Aucun fichier audio ou transcript ne doit être envoyé vers OpenAI, Mistral, Microsoft ou un autre service externe.

Les téléchargements initiaux des modèles Hugging Face sont évidemment autorisés.

Une fois les modèles présents localement, le traitement des réunions doit pouvoir s'effectuer sans envoi du contenu audio à un service distant.

---

# 27. Robustesse

Traiter proprement :

* fichier vidéo sans piste audio ;
* fichier audio corrompu ;
* Ollama arrêté ;
* aucun modèle Ollama installé ;
* token Hugging Face manquant ;
* CUDA indisponible ;
* erreur CUDA out-of-memory ;
* diarisation ne trouvant qu'un speaker ;
* nombre de speakers détecté différent du nombre de participants ;
* transcript vide ;
* nom de fichier contenant accents ou espaces.

En cas de CUDA OOM :

1. libérer les modèles inutilisés ;
2. vider le cache CUDA ;
3. afficher une erreur claire ;
4. ne pas perdre les fichiers déjà produits.

---

# 28. Tests

Les fonctions métier qui ne nécessitent pas réellement les modèles doivent avoir des tests unitaires.

Tester au minimum :

### Alignement

Cas :

```text
ASR segment : 10 → 20 s

speaker A : 10 → 14
speaker B : 14 → 20
```

Résultat attendu :

```text
speaker B
```

si la règle est la durée de recouvrement maximale.

### Mapping

Tester :

```text
SPEAKER_00 → Vincent
```

sur tout le transcript.

### Fusion

Tester la fusion de plusieurs segments consécutifs du même speaker.

### Export JSON/Markdown

Vérifier timestamps et encodage UTF-8.

### Ollama

Mock de l'API pour :

* liste des modèles ;
* génération ;
* serveur inaccessible.

---

# 29. Critères d'acceptation V1

La V1 est considérée comme terminée si le scénario suivant fonctionne sur la machine réelle :

1. lancer :

```bash
streamlit run app.py
```

2. ouvrir l'application dans le navigateur ;

3. uploader un MP4 de réunion ;

4. saisir :

```text
Vincent
Charles
```

5. lancer le traitement ;

6. extraire correctement l'audio ;

7. effectuer la transcription avec Parakeet sur GPU ;

8. effectuer la diarisation avec pyannote ;

9. obtenir deux speakers distincts ;

10. proposer leur association avec Vincent et Charles ;

11. permettre à l'utilisateur de corriger manuellement cette association ;

12. produire une transcription du type :

```text
[00:01:04] Vincent
...

[00:01:12] Charles
...
```

13. modifier le prompt du compte-rendu ;

14. sélectionner un modèle Ollama installé ;

15. générer le compte-rendu ;

16. télécharger :

```text
MP3
transcript
compte-rendu
ZIP complet
```

---

# 30. Méthode de développement demandée

Ne pas écrire toute l'application d'un seul coup sans la tester.

Procéder incrémentalement :

### Étape 1

Inspecter l'environnement :

```bash
python --version
nvidia-smi
ffmpeg -version
ollama --version
ollama list
```

et vérifier PyTorch/CUDA.

### Étape 2

Créer et tester isolément la transcription Parakeet sur un petit fichier audio.

### Étape 3

Créer et tester isolément pyannote sur le même fichier.

### Étape 4

Implémenter et tester l'alignement ASR/diarisation.

### Étape 5

Tester Ollama depuis Python.

### Étape 6

Créer ensuite l'interface Streamlit.

### Étape 7

Tester le workflow complet avec un vrai fichier de réunion.

À chaque étape, corriger les problèmes de dépendances ou de mémoire avant de continuer.

Ne pas simplement produire du code supposé fonctionner : exécuter les tests et les commandes sur la machine lorsque cela est possible.

---

# 31. Référence existante

Un ancien script Python `extract_audio.py` existe.

Il sert uniquement de référence pour comprendre le workflow métier existant :

```text
fichier audio/vidéo
→ extraction audio
→ transcription
→ compte-rendu
```

Ne pas conserver son architecture monolithique.

Ne pas conserver :

```text
OpenAI whisper-1
GPT-4o
découpage arbitraire en fichiers segment_*.mp3
```

La nouvelle application doit remplacer ce fonctionnement par l'architecture locale décrite ci-dessus.

---

# 32. Priorités

En cas d'arbitrage, respecter cet ordre :

1. fiabilité de la transcription ;
2. bonne attribution des speakers ;
3. stabilité sur GPU 6 Go ;
4. conservation de tous les résultats intermédiaires ;
5. simplicité d'utilisation ;
6. qualité du compte-rendu ;
7. performances ;
8. sophistication visuelle.

L'interface peut rester sobre.

La priorité est d'obtenir un outil réellement fiable pour traiter des réunions.
