import os
import re
import time
import json
import logging
import shutil
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import mutagen  

load_dotenv()

# Initialize structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

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
    "आप एक शब्द-ब-शब्द (literal, word-for-word) ऑडियो ट्रांसक्राइबर हैं। एक पूरी तरह से निष्क्रिय श्रोता (passive listener) के रूप में कार्य करें जो केवल वही लिखता है जो सुना जाता है। पाठ (text) को \"ठीक\" या सुचारू करने के लिए किसी भी बाहरी ज्ञान, बुद्धिमत्ता या संदर्भ (context) का उपयोग न करें。\n\n"
    "**सख्त नियम (Strict Rules):**\n\n"
    "1. **लहजा और बोलियाँ (Accents & Dialects):** वक्ता (speaker) का लहजा भारतीय (Indian accent) है। बोले गए हिंदी शब्दों को बिल्कुल वैसे ही ट्रांसक्राइब करें जैसे वे उच्चारित किए जा रहे हैं।\n"
    "2. **खराब गुणवत्ता वाला ऑडियो (Poor Quality Audio):** यदि कोई शब्द दबा हुआ (muffled), विकृत (distorted), या कटा हुआ है, तो अनुमान न लगाएं कि वक्ता क्या कहना चाहता था। जो टूटा-फूटा ध्वन्यात्मक शब्द (phonetic sound) सुनाई दे, उसे बिल्कुल वैसा ही लिखें, या यदि वह पूरी तरह से अस्पष्ट हो तो [unclear] का उपयोग करें।\n"
    "3. **कोई ऑटो-करेक्ट नहीं (No Autocorrect):** व्याकरण (grammar) को न सुधारें, टूटी हुई वाक्य संरचनाओं को न बदलें, और स्लैंग (slang) या बोलचाल की भाषा को मानक हिंदी से न बदलें।\n"
    "4. **अक्षरशः अनुवाद (Literalism):** यदि वक्ता किसी शब्द को दोहराता है, हकलाता है, या बोलने में कोई चूक करता है, तो उसे बिल्कुल वैसा ही ट्रांसक्राइब करें। इसे हटाएँ या संपादित (edit) न करें।\n"
    "5. **कोई अतिरिक्त जानकारी नहीं (No Additions):** कोई परिचय, स्पष्टीकरण, सारांश या नोट्स न जोड़ें। आउटपुट में केवल और केवल ट्रांसक्राइब किया गया टेक्स्ट ही होना चाहिए。\n"
    "Transcribe the attached audio file, which is in Hindi with some Sanskrit phrases."
)

def get_public_drive_service():
    sa_json_str = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    if not sa_json_str:
        raise ValueError("❌ Missing GDRIVE_SERVICE_ACCOUNT in environment variables!")
    sa_info = json.loads(sa_json_str)
    credentials = service_account.Credentials.from_service_account_info(sa_info)
    return build('drive', 'v3', credentials=credentials)

def get_audio_duration(file_path):
    """Auto-detects audio format (.mp3, .wav, .m4a, .flac, etc.) and returns formatted duration in minutes."""
    try:
        audio = mutagen.File(file_path)
        if audio is not None and audio.info is not None:
            mins = audio.info.length / 60
            return f"{mins:.2f} minutes"
        return "Unknown duration"
    except Exception as e:
        logging.warning(f"⚠️ Could not extract audio duration: {e}")
        return "Unknown duration"

def update_google_drive_file(drive_service, file_id, local_file_path, drive_filename):
    """Overwrites an existing user-owned placeholder file on Google Drive using its file ID.
       Protects against files <1KB and completely prevents Broken Pipe errors using resumable chunks and retries."""
    try:
        # GUARD: Check size before touching Google Drive
        file_size = os.path.getsize(local_file_path)
        if file_size < 1024:
            logging.warning(f"⚠️ Guard Skipped: Local transcript '{drive_filename}' size ({file_size} bytes) is < 1KB. Google Drive file will NOT be overwritten.")
            return False
    except Exception as e:
        logging.error(f"❌ File validation reading failed: {e}")
        return False

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # FIX: resumable=True mitigates Broken Pipe errors by chunking transmissions
            media = MediaFileUpload(local_file_path, mimetype='text/plain', resumable=True)
            updated_file = drive_service.files().update(
                fileId=file_id,
                media_body=media,
                fields='id'
            ).execute()
            logging.info(f"☁️ Google Drive placeholder updated! (File ID: {updated_file.get('id')})")
            return True
        except Exception as e:
            attempt += 1
            logging.warning(f"⚠️ Upload Attempt {attempt}/{MAX_RETRIES} encountered an error: {e}")
            if attempt < MAX_RETRIES:
                sleep_cooldown = 2 ** attempt * 5
                logging.info(f"Cooling down for {sleep_cooldown} seconds before retry...")
                time.sleep(sleep_cooldown)
            else:
                logging.error(f"❌ CRITICAL: Google Drive Overwrite failed persistently after {MAX_RETRIES} attempts.")
                return False

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    logging.info("🎬 Starting Streamlined Public Pipeline...")
    
    # Clean workspace startup sequence
    if os.path.exists(AUDIO_FOLDER):
        shutil.rmtree(AUDIO_FOLDER)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

    logging.info("📥 Scanning Google Drive target folder...")
    try:
        drive_service = get_public_drive_service()
        items = []
        page_token = None
        while True:
            results = drive_service.files().list(
                q=f"'{FOLDER_ID}' in parents and trashed=false", 
                fields="nextPageToken, files(id, name, size)",
                pageToken=page_token,
                pageSize=1000
            ).execute()
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        logging.error(f"❌ Drive Folder Scan Failed: {e}.")
        return

    local_transcripts = set(os.listdir(TRANSCRIPT_FOLDER)) if os.path.exists(TRANSCRIPT_FOLDER) else set()
    drive_file_map = {item['name']: item for item in items}
    pending_audio = []
    
    for item in items:
        if item['name'].lower().endswith(SUPPORTED_FORMATS):
            base_name, _ = os.path.splitext(item['name'])
            transcript_name = f"{base_name}_transcript.txt"
            
            if transcript_name in local_transcripts:
                logging.info(f"⏩ Skipping {item['name']} (Already transcribed in GitHub repository)")
                continue
            
            if transcript_name in drive_file_map:
                drive_file = drive_file_map[transcript_name]
                if 'size' in drive_file and int(drive_file['size']) > 1024:
                    logging.info(f"⏩ Skipping {item['name']} (Already transcribed in Google Drive)")
                    continue
                    
            pending_audio.append(item) 

    pending_audio.sort(key=lambda x: natural_sort_key(x['name']))

    if not pending_audio:
        logging.info("ℹ️ No new audio files to process.")
        return

    logging.info(f"🚀 Found {len(pending_audio)} pending tracks. Initializing Gemini Client...")
    
    # Explicit Key allocation for strict reliability parameters
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
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

        logging.info(f"📥 [{i}/{len(pending_audio)}] Downloading: {filename}...")
        try:
            request = drive_service.files().get_media(fileId=file_id)
            with open(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
        except Exception as e:
            logging.error(f"❌ Drive API Download failed or timed out: {e}")
            continue

        audio_duration = get_audio_duration(file_path)
        success = False
        audio_upload = None
        file_start_time = time.time()
        network_retries = 0
        rate_limit_retries = 0

        while not success:
            try:
                if not audio_upload:
                    logging.info("⏳ Uploading file to Gemini File storage API...")
                    audio_upload = client.files.upload(file=file_path)
                    
                file_info = client.files.get(name=audio_upload.name)
                
                # Adaptive Polling: Scales down frequency slowly to preserve rate limits
                poll_interval = 5
                while "processing" in str(file_info.state).lower():
                    time.sleep(poll_interval)
                    file_info = client.files.get(name=audio_upload.name)
                    poll_interval = min(20, poll_interval + 2)

                if "failed" in str(file_info.state).lower():
                    logging.error("❌ Error: Google cloud audio processing failed inside the file API.")
                    break

                logging.info(f"-> Processing transcription via {MODEL_ID}...")
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[audio_upload, "Please transcribe this audio file verbatim."],
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
                )

                if not response.text or not response.text.strip():
                    raise RuntimeError("Gemini returned an empty or invalid transcript context.")

                transcript_body = response.text
                header_text = (
                    f"File name: {filename}\n"
                    f"Audio file duration: {audio_duration}\n"
                    f"----------------------------------------\n\n"
                )

                # Local Save Pipeline
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(header_text + transcript_body)
                logging.info(f"✅ Saved Transcript Locally: {transcript_filename} (⏱️ {(time.time() - file_start_time)/60:.2f}m)")

                # Structural size safety guard check before committing resources
                if os.path.getsize(output_path) <= 1024:
                    logging.warning(f"⚠️ Alert: Generated transcript file is <= 1KB. Dropping local assets.")
                    try: os.remove(output_path)
                    except Exception: pass
                    try: os.remove(file_path)
                    except Exception: pass
                    break

                # Cloud Synced Update Pipeline
                if transcript_filename in drive_file_map:
                    placeholder_id = drive_file_map[transcript_filename]['id']
                    update_google_drive_file(drive_service, placeholder_id, output_path, transcript_filename)
                else:
                    logging.warning(f"⚠️ WARNING: No pre-created placeholder found on Drive for '{transcript_filename}'. Skipping Drive sync.")

                success = True
                try: os.remove(file_path) 
                except Exception: pass
                time.sleep(3)

            except APIError as e:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES:
                    logging.error("❌ CRITICAL: Generation limits hit. Terminating remaining queue execution loop.")
                    quota_exhausted = True  
                    try: os.remove(file_path)
                    except Exception: pass
                    break 
                
                # Exponential Backoff Calculations
                sleep_time = min(300, 2 ** rate_limit_retries * 30)
                logging.warning(f"⚠️ Rate limit hit. Cooling down for {sleep_time}s...")
                time.sleep(sleep_time)

            except Exception as e: 
                error_msg = str(e).lower()
                if "timeout" in error_msg or "deadline" in error_msg:
                    logging.error("❌ Execution Stalled: File took over 20 minutes to process. Aborting this track.")
                    try: os.remove(file_path)
                    except Exception: pass
                    success = True 
                    break
                
                if any(x in error_msg for x in ["disconnected", "connection", "eof"]):
                    network_retries += 1
                    if network_retries > MAX_RETRIES:
                        logging.warning("⚠️ Persistent dropped connections. Skipping execution frame for this track.")
                        try: os.remove(file_path)
                        except Exception: pass
                        success = True  
                        break            
                    time.sleep(60 * network_retries)
                else:
                    logging.error(f"❌ Unexpected Pipeline Exception: {e}")
                    try: os.remove(file_path)
                    except Exception: pass
                    success = True 
                    break
            finally:
                if success and audio_upload:
                    try: client.files.delete(name=audio_upload.name)
                    except Exception: pass

    logging.info("🏁 All operations completed successfully! Transcripts folder updated.")

if __name__ == "__main__":
    main()
