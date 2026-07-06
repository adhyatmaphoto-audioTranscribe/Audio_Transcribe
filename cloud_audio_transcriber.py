import os
import re
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload  
import requests
from mutagen.mp3 import MP3  

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = 'gemini-3.5-flash' 
MAX_RETRIES = 2  
FOLDER_ID = '17MGWbLC8Qq_UxLCtOePc9aJgVSAakhde'         

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(SCRIPT_DIR, 'temp_audio')       
TRANSCRIPT_FOLDER = os.path.join(SCRIPT_DIR, 'transcripts') 
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
    """Builds a Drive service reader/writer using the Service Account string from environment memory."""
    sa_json_str = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    if not sa_json_str:
        raise ValueError("❌ Missing GDRIVE_SERVICE_ACCOUNT in environment variables!")
    sa_info = json.loads(sa_json_str)
    credentials = service_account.Credentials.from_service_account_info(sa_info)
    return build('drive', 'v3', credentials=credentials)

def get_mp3_duration(file_path):
    try:
        audio = MP3(file_path)
        mins = audio.info.length / 60
        return f"{mins:.2f} minutes"
    except Exception:
        return "Unknown duration"

def upload_to_google_drive(drive_service, local_file_path, drive_filename, parent_folder_id):
    """Uploads the completed transcript text file directly back to your Google Drive folder instantly.
       If it fails, catches the error, flashes a warning, and returns safely without crashing."""
    try:
        file_metadata = {
            'name': drive_filename,
            'parents': [parent_folder_id]
        }
        media = MediaFileUpload(local_file_path, mimetype='text/plain')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"   ☁️ Successfully uploaded to Google Drive! (File ID: {uploaded_file.get('id')})")
        return True
    except Exception as e:
        print(f"   ⚠️ WARNING: Google Drive Upload Failed for '{drive_filename}': {e}")
        print("   ⚠️ Script will continue processing remaining pipeline operations.")
        return False

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    print("🎬 Starting Streamlined Public Pipeline...")
    
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

    print("\n📥 Scanning Google Drive target folder...")
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
        print(f"❌ Drive Folder Scan Failed: {e}.")
        return

    # Gather local GitHub transcript files
    local_transcripts = set(os.listdir(TRANSCRIPT_FOLDER)) if os.path.exists(TRANSCRIPT_FOLDER) else set()
    
    # Gather Google Drive inventory filenames
    drive_filenames = {item['name'] for item in items}
    
    pending_audio = []
    
    for item in items:
        if item['name'].lower().endswith(SUPPORTED_FORMATS):
            base_name, _ = os.path.splitext(item['name'])
            transcript_name = f"{base_name}_transcript.txt"
            
            # DUAL CHECK: Check if transcript exists locally ON GITHUB *OR* UPSTAIRS ON GOOGLE DRIVE
            if transcript_name in local_transcripts or transcript_name in drive_filenames:
                print(f"⏩ Skipping {item['name']} (Already transcribed in GitHub repository or Google Drive)")
                continue
            pending_audio.append(item) 

    pending_audio.sort(key=lambda x: natural_sort_key(x['name']))

    if not pending_audio:
        print(f"ℹ️ No new audio files to process.")
        return

    print(f"🚀 Found {len(pending_audio)} pending tracks. Initializing Gemini Client with safety timeouts...")
    
    # Global HTTP timeout configuration: Kills silent generation sockets if they hang over 20 minutes
    client = genai.Client(
        http_options=types.HttpOptions(timeout=20 * 60 * 1000)
    ) 
    quota_exhausted = False

    for i, item in enumerate(pending_audio, 1):
        if quota_exhausted:
            break

        filename = item['name']
        file_id = item['id']
        file_path = os.path.join(AUDIO_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)
        transcript_filename = f"{base_name}_transcript.txt"
        output_path = os.path.join(TRANSCRIPT_FOLDER, transcript_filename)

        print(f"\n📥 [{i}/{len(pending_audio)}] Downloading: {filename}...")
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url, stream=True, timeout=300)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"   ❌ Stream Download failed or timed out: {e}")
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
                    print(f"⏳ Uploading file to Gemini File storage API...")
                    audio_upload = client.files.upload(file=file_path)
                    
                file_info = client.files.get(name=audio_upload.name)
                while "processing" in str(file_info.state).lower():
                    time.sleep(5)
                    file_info = client.files.get(name=audio_upload.name)

                if "failed" in str(file_info.state).lower():
                    print("   ❌ Error: Google cloud audio processing failed inside the file API.")
                    break

                print(f"   -> Processing transcription via {MODEL_ID}...")
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[audio_upload, "Please transcribe this audio file verbatim."],
                    config=types.GenerateContentConfig(temperature=0.0)
                )

                transcript_body = response.text or "[Empty transcript returned]"
                header_text = (
                    f"File name: {filename}\n"
                    f"Audio file duration: {audio_duration}\n"
                    f"----------------------------------------\n\n"
                )

                # Step 1: Save local backup copy on the runner's workspace disk
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(header_text + transcript_body)
                print(f"   ✅ Saved Transcript Locally: {transcript_filename} (⏱️ {(time.time() - file_start_time)/60:.2f}m)")

                # Step 2: Push back up to the Google Drive folder instantly (Safe from crashing)
                upload_to_google_drive(drive_service, output_path, transcript_filename, FOLDER_ID)

                # Run structural size safety checks
                if os.path.getsize(output_path) <= 1024:
                    print(f"   ⚠️ Alert: File content is less than or equal to 1KB. Dropping resource.")
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
                    print(f"\n❌ CRITICAL: Generation limits hit. Terminating remaining queue execution loop.")
                    quota_exhausted = True  
                    try: os.remove(file_path)
                    except Exception: pass
                    break 
                
                print(f"\n⚠️ Rate limit hit. Cooling down for {60 * rate_limit_retries}s...")
                time.sleep(60 * rate_limit_retries)

            except Exception as e: 
                error_msg = str(e).lower()
                if "timeout" in error_msg or "deadline" in error_msg:
                    print(f"   ❌ Execution Stalled: File took over 20 minutes to process. Aborting this track.")
                    try: os.remove(file_path)
                    except Exception: pass
                    success = True 
                    break
                
                if any(x in error_msg for x in ["disconnected", "connection", "eof"]):
                    network_retries += 1
                    if network_retries > MAX_RETRIES:
                        print(f"\n⚠️ Persistent dropped connections. Skipping execution frame for this track.")
                        try: os.remove(file_path)
                        except Exception: pass
                        success = True  
                        break            
                    time.sleep(60 * network_retries)
                else:
                    print(f"   ❌ Unexpected Pipeline Exception: {e}")
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
