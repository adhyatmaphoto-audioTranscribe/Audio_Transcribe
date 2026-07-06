import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from googleapiclient.discovery import build
import requests
from mutagen.mp3 import MP3  # Requires: pip install mutagen

# Load your Gemini API Key safely from your local .env file
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = 'gemini-3.5-flash' 
MAX_RETRIES = 2  

# UPDATED: Google Drive Public Folder ID extracted from your link
FOLDER_ID = '17MGWbLC8Qq_UxLCtOePc9aJgVSAakhde'         

# CLEAN DIRECTORY SEPARATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(SCRIPT_DIR, 'temp_audio')       # <-- Add this folder to your .gitignore
TRANSCRIPT_FOLDER = os.path.join(SCRIPT_DIR, 'transcripts') # <-- DO NOT ignore this folder (Saves TXT files to GitHub)
SUPPORTED_FORMATS = ('.mp3', '.wav', '.m4a', '.aac', '.flac')

SYSTEM_PROMPT = (
    "You are a precise, machine-level verbatim transcriptionist.\n"
    "Transcribe the provided audio file exactly as spoken, word-for-word. "
    "Do not summarize, condense, or edit. Output the literal transcript text only.\n\n"
    "Strict Rules:\n"
    "1. Multi-Lingual & Multi-Script Handling:\n"
    "   - The primary audio is English spoken with an Indian (Odisha region) accent.\n"
    "   - For Sanskrit and Hindi phrases: Transcribe them strictly in Devanagari script.\n"
    "   - For Odia (Oriya) and Bengali phrases: If you are 100% certain of the language, transcribe them in their respective native scripts. Default to Devanagari if uncertain.\n"
    "   - Do NOT translate any non-English phrases into English.\n"
    "2. Poor Quality Audio: Write exactly the broken phonetic sound heard, or use [unclear].\n"
    "3. No Autocorrect: Do not correct grammar or broken sentence structures.\n"
    "4. Literalism: Transcribe stutters or repetitions exactly.\n"
    "5. Formatting Constraints: Output ONLY the raw transcript text."
)

def get_public_drive_service():
    """Builds a public Drive reader using ONLY your API Key."""
    api_key = os.environ.get("GEMINI_API_KEY")
    return build('drive', 'v3', developerKey=api_key)

def get_mp3_duration(file_path):
    """Reads the true duration of the MP3 file directly from disk like Windows File Manager."""
    try:
        audio = MP3(file_path)
        mins = audio.info.length / 60
        return f"{mins:.2f} minutes"
    except Exception:
        return "Unknown duration"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    print("🎬 Starting Streamlined Public Pipeline...")
    
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

    # 1. SCAN PUBLIC FOLDER USING API KEY
    print("\n📥 Scanning public Google Drive folder...")
    try:
        drive_service = get_public_drive_service()
        items = []
        page_token = None
        while True:
            results = drive_service.files().list(
                q=f"'{FOLDER_ID}' in parents and trashed=false", 
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                pageSize=1000
            ).execute()
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        print(f"❌ Public Drive Scan Failed: {e}. Check if folder link sharing is active.")
        return

    local_transcripts = os.listdir(TRANSCRIPT_FOLDER)
    pending_audio = []
    
    for item in items:
        if item['name'].lower().endswith(SUPPORTED_FORMATS):
            base_name, _ = os.path.splitext(item['name'])
            if f"{base_name}_transcript.txt" in local_transcripts:
                print(f"⏩ Skipping {item['name']} (Already transcribed matching text file)")
                continue
            pending_audio.append(item) 

    pending_audio.sort(key=lambda x: natural_sort_key(x['name']))

    if not pending_audio:
        print(f"ℹ️ No new audio files to process.")
        return

    print(f"🚀 Found {len(pending_audio)} pending tracks. Initializing Gemini...")
    client = genai.Client() 
    quota_exhausted = False

    # 2. RUN TRANSCRIPTION BATCH
    for i, item in enumerate(pending_audio, 1):
        if quota_exhausted:
            break

        filename = item['name']
        file_id = item['id']
        file_path = os.path.join(AUDIO_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(TRANSCRIPT_FOLDER, f"{base_name}_transcript.txt")

        print(f"\n📥 [{i}/{len(pending_audio)}] Downloading via Public Stream: {filename}...")
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            continue

        audio_duration = get_mp3_duration(file_path)

        success = False
        audio_upload = None
        file_start_time = time.time()
        network_retries = 0
        rate_limit_retries = 0

        while not success:
            try:
                if not audio_upload:
                    print(f"⏳ Uploading 30MB file to Gemini File storage API...")
                    audio_upload = client.files.upload(file=file_path)
                    
                file_info = client.files.get(name=audio_upload.name)
                while "processing" in str(file_info.state).lower():
                    time.sleep(5)
                    file_info = client.files.get(name=audio_upload.name)

                if "failed" in str(file_info.state).lower():
                    print("   ❌ Error: Google cloud audio processing failed.")
                    break

                print(f"   -> Processing transcription via {MODEL_ID}...")
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[audio_upload, "Please transcribe this audio file verbatim."],
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
                )

                transcript_body = response.text or "[Empty transcript returned]"
                header_text = (
                    f"File name: {filename}\n"
                    f"Audio file duration: {audio_duration}\n"
                    f"----------------------------------------\n\n"
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(header_text + transcript_body)
                print(f"   ✅ Saved Transcript: {base_name}_transcript.txt (⏱️ {(time.time() - file_start_time)/60:.2f}m)")

                if os.path.getsize(output_path) <= 1024:
                    print(f"   ⚠️ Alert: File <= 1KB. Executing structural drop rule.")
                    try: os.remove(output_path)
                    except Exception: pass
                    try: os.remove(file_path)
                    except Exception: pass
                    break

                success = True
                try: os.remove(file_path) 
                except Exception: pass
                time.sleep(3)

            except APIError as e:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES:
                    print(f"\n❌ CRITICAL: Quota exhausted. Aborting queue processing run.")
                    quota_exhausted = True  
                    try: os.remove(file_path)
                    except Exception: pass
                    break 
                
                print(f"\n⚠️ Rate limit hit. Pausing for {60 * rate_limit_retries}s...")
                time.sleep(60 * rate_limit_retries)

            except Exception as e: 
                error_msg = str(e).lower()
                if any(x in error_msg for x in ["disconnected", "connection", "eof"]):
                    network_retries += 1
                    if network_retries > MAX_RETRIES:
                        print(f"\n⚠️ Persistent connection drops. Skipping track.")
                        try: os.remove(file_path)
                        except Exception: pass
                        success = True  
                        break            
                    time.sleep(60 * network_retries)
                else:
                    print(f"   ❌ Unexpected Error: {e}")
                    try: os.remove(file_path)
                    except Exception: pass
                    success = True 
                    break
            finally:
                if success and audio_upload:
                    try: client.files.delete(name=audio_upload.name)
                    except Exception: pass

    print("\n🏁 All operations completed successfully! Transcripts folder updated.")

if __name__ == "__main__":
    main()
