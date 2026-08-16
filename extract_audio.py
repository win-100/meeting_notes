from openai import OpenAI
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import moviepy as mp

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

client = OpenAI()

audio_path = "C:/Users/win10/Videos/2026-07-31 11-00-24 - Chaussea - revue lettre de cadrage.mp4"
minute_additional_prompt = "" # "Absolutely everywhere you read \"AstorApp\", please understand it as la Store App. "

main_prompt = ("Voici ci-après la transcription complète de mon entretien avec Charles, directeur digital de Chaussea. "
    "C'est un entretien pour revoir ensemble les commentaires que j'ai mis sur la lettre de cadrage de mission qu'il a rédigée. " 
    "Il s'agit d'une mission de chef de projet pour l'implémentation de l'OMS OneStock chez Chaussea. Le projet est déjà démarré et il faudrait faire un audit de la situation avant de reprendre le projet. "
    "C'est une transcription brute, il peut y avoir des erreurs de transcription, et les interlocuteurs ne sont pas identifiés. "
    "Il faut donc déduire qui a parlé entre moi (Vincent) et Charles. "
    "PeoPulse ou d'autres noms sont peut-être mal prononcés, il faut les corriger si la transcription est mauvaise. On évoque aussi fréquemment mon ancienne entreprise OneStock. "
    "Je voudrais que tu m'écrives des notes de cet entretien, pour moi, Vincent sans détailler mon parcours puisque je le connais déjà. "
    "Le compte-rendu doit être rédigé en français, en markdown, avec un seul titre de niveau 1 puis chaque section en titre de niveau 2. "
    "Ce doit être concis, bien structuré et couvrir tous les sujets traités mais pas forcément dans l'ordre chronologique. "
)

# Définir le chemin de sortie pour le fichier de transcription
audio_base_name = os.path.splitext(audio_path)[0]
output_text_path = f"{audio_base_name}.txt"
minute_text_path = f"{audio_base_name}_minute.txt"

# Définir la taille maximale en octets (25 Mo)
MAX_CONTENT_SIZE = 25 * 1024 * 1024

def extraire_audio(mp4_file_path):
    base_name = os.path.splitext(mp4_file_path)[0]
    audio_output_path = f"{base_name}.mp3"
    print("Extraction de l'audio en cours...")
    video = mp.VideoFileClip(mp4_file_path)
    video.audio.write_audiofile(audio_output_path)
    print("Extraction de l'audio terminée.")
    return audio_output_path


def diviser_audio(mp3_file_path, segment_duration_ms=300000):
    audio = AudioSegment.from_mp3(mp3_file_path)
    duration_ms = len(audio)
    segments = []
    for i in range(0, duration_ms, segment_duration_ms):
        segment = audio[i:i + segment_duration_ms]
        segments.append(segment)
    return segments, duration_ms


def transcrire_audio_segment(segment, segment_index, total_segments):
    try:
        segment_path = f"segment_{segment_index}.mp3"
        segment.export(segment_path, format="mp3")
        
        # Charger le fichier audio segmenté
        with open(segment_path, "rb") as audio_file:
            # Utiliser l'API OpenAI Whisper pour transcrire l'audio
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        progress = (segment_index + 1) / total_segments * 100
        print(f"Transcription du segment {segment_index + 1}/{total_segments} terminée ({progress:.2f}% complet)")
        return response.text
    except Exception as e:
        print(f"Erreur lors de la transcription du segment {segment_index} : {e}")
        return None
    finally:
        if os.path.exists(segment_path):
            os.remove(segment_path)


def creer_compte_rendu(transcription):
    try:
        prompt = main_prompt + minute_additional_prompt
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who writes professional meeting minutes."},
                {"role": "user", "content": f"{prompt}\n\n{transcription}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Erreur lors de la création du compte-rendu : {e}")
        return None


if __name__ == "__main__":
    # Vérifier le type de fichier et exécuter les étapes appropriées
    transcription_complete = ""
    
    if audio_path.endswith(".mp4"):
        # Extraire l'audio d'une vidéo
        audio_path = extraire_audio(audio_path)
    
    if audio_path.endswith(".mp3"):
        # Diviser le fichier audio en segments et transcrire
        segments, duration_ms = diviser_audio(audio_path)
        total_segments = len(segments)
        
        print("Début de la transcription...")
        for index, segment in enumerate(segments):
            transcription_segment = transcrire_audio_segment(segment, index, total_segments)
            if transcription_segment:
                transcription_complete += transcription_segment + "\n"
        
        if transcription_complete:
            # Sauvegarder la transcription complète dans un fichier texte
            with open(output_text_path, "w", encoding="utf-8") as fichier_sortie:
                fichier_sortie.write(transcription_complete)
            print(f"Transcription complète enregistrée dans '{output_text_path}'.")
        else:
            print("La transcription a échoué.")
    
    if audio_path.endswith(".txt") or transcription_complete:
        # Si un fichier texte est passé ou si la transcription est complète, créer un compte-rendu de réunion
        if not transcription_complete:
            with open(audio_path, "r", encoding="utf-8") as fichier_transcription:
                transcription_complete = fichier_transcription.read()
        
        # Créer un compte-rendu de la réunion
        print("Création du compte-rendu de la réunion...")
        compte_rendu = creer_compte_rendu(transcription_complete)
        if compte_rendu:
            with open(minute_text_path, "w", encoding="utf-8") as fichier_compte_rendu:
                fichier_compte_rendu.write(compte_rendu)
            print(f"Compte-rendu de la réunion enregistré dans '{minute_text_path}'.")
        else:
            print("La création du compte-rendu a échoué.")
    else:
        print("Type de fichier non supporté. Veuillez fournir un fichier .mp3, .mp4, ou .txt.")