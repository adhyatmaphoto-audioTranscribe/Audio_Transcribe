import os
import io
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Google Cloud Drive API Modules
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# =============================================================================
# SECURE CONFIGURATION CONTROL PANEL
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
DRIVE_TOKEN_FILE = 'token.json'

MODEL_ID = 'gemini-3.5-flash'
MAX_RETRIES = 2  
SUPPORTED_FORMATS = ('.mp3', '.wav', '.m4a', '.aac', '.flac')

SYSTEM_PROMPT = (
    "You are a precise, machine-level verbatim transcriptionist.\n"
    "Transcribe the provided audio file exactly as spoken, word-for-word. "
    "Do not summarize, condense, or edit. Output the literal transcript text only.\n\n"
    "Strict Rules:\n"
    "1. Multi-Lingual & Multi-Script Handling:\n"
    "   - The primary audio is English spoken with an Indian (Odisha region) accent.\n"
    "   - For Sanskrit and Hindi phrases: Transcribe them strictly in the Devanagari script.\n"
    "   - For Odia (Oriya) and Bengali phrases: If you are 100% certain of the language, transcribe them in their respective native scripts (Odia/Bengali script). If you are uncertain, default to transcribing those Odia and Bengali phrases in Devanagari script.\n"
    "   - Do NOT translate any non-English phrases into English. Transcribe the spoken sounds into the designated script.\n"
    "2. Poor Quality Audio: If a word is muffled, distorted, or cut off, do not guess what the speaker meant to say. Write exactly the broken phonetic sound heard, or use [unclear] if it is entirely unintelligible.\n"
    "3. No Autocorrect: Do not correct grammar, do not change broken sentence structures, and do not replace slang or colloquialisms.\n"
    "4. Literalism: If the speaker repeats a word, stutters, or makes a verbal slip, transcribe it exactly. Do not edit it out.\n"
    "5. Formatting Constraints:\n"
    "   - Do not add conversational introductions, explanations, summaries, or metadata notes.\n"
    "   - Output ONLY the raw transcript text."
)

# =============================================================================
# API CORE INITIALIZERS
# =============================================================================
def get_drive_service():
    """Authenticates with the Google Drive API via Personal User OAuth Token."""
    if not os.path.exists(DRIVE_TOKEN_FILE):
        raise FileNotFoundError(f"Missing '{DRIVE_TOKEN_FILE}' in cloud context.")
    
    creds = Credentials.from_authorized_user_file(DRIVE_TOKEN_FILE, scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=creds)

def get_drive_files_dict(service, folder_id):
    """Fetches a map of all files inside a target Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {file['name']: file['id'] for file in results.get('files', [])}

# =============================================================================
# MAIN AUTOMATION PIPELINE
# =============================================================================
def run_cloud_batch_transcription(input_folder_id, output_folder_id):
    drive_service = get_drive_service()
    
    print("📡 Fetching file structures from Google Drive Input/Output Folders...")
    input_drive_files = get_drive_files_dict(drive_service, input_folder_id)
    output_drive_files = get_drive_files_dict(drive_service, output_folder_id)

    # Filter for supported tracks matching extensions
    tracks_to_process = {name: fid for name, fid in input_drive_files.items() if name.lower().endswith(SUPPORTED_FORMATS)}
    
    if not tracks_to_process:
        print("ℹ️ No supported audio files found in the source Google Drive folder.")
        return

    print(f"🚀 Found {len(tracks_to_process)} target tracks. Initializing Gemini Client runtime...")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=120000))

    for i, (filename, file_id) in enumerate(tracks_to_process.items(), 1):
        base_name, _ = os.path.splitext(filename)
        transcript_name = f"{base_name}_transcript.txt"

        # Check duplication constraint directly in cloud output directory
        if transcript_name in output_drive_files:
            print(f"⏩ [{i}/{len(tracks_to_process)}] Skipping '{filename}' (Transcript already exists on Drive).")
            continue

        temp_local_audio = f"temp_{filename}"
        success = False
        audio_upload = None
        file_start_time = time.time()
        backoff_attempt = 0

        # Download targeted file stream into active runner disk space
        print(f"⏳ [{i}/{len(tracks_to_process)}] Downloading '{filename}' from Drive to runner...")
        request = drive_service.files().get_media(fileId=file_id)
        with open(temp_local_audio, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        while not success:
            try:
                if not audio_upload:
                    print(f"   -> Uploading to Gemini Storage cloud engine...")
                    upload_start = time.time()
                    audio_upload = gemini_client.files.upload(file=temp_local_audio)
                    print(f"   -> Upload complete in {time.time() - upload_start:.1f} seconds.")

                file_info = gemini_client.files.get(name=audio_upload.name)
                while "processing" in str(file_info.state).lower():
                    print("      [...] Backend processing metadata structures. Waiting 5 seconds...")
                    time.sleep(5)
                    file_info = gemini_client.files.get(name=audio_upload.name)

                if "failed" in str(file_info.state).lower():
                    print("   ❌ Error: Google cloud pipeline extraction failed for this file.")
                    break

                print(f"   -> Processing via {MODEL_ID} inference engines...")
                inference_start = time.time()

                response = gemini_client.models.generate_content(
                    model=MODEL_ID,
                    contents=[audio_upload, "Please transcribe this audio file verbatim."],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0
                    )
                )

                print(f"   -> Server returned response in {time.time() - inference_start:.1f} seconds.")
                transcript_body = response.text
                
                if not transcript_body:
                    try:
                        if response.candidates and response.candidates[0].content.parts:
                            transcript_body = response.candidates[0].content.parts[0].text
                    except Exception: pass

                if not transcript_body:
                    transcript_body = "[Error: Empty stream packet returned from server.]"

                # Calculate duration approximation from token data payload
                header_text = f"File name: {filename}\n"
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    approx_minutes = (input_tokens / 32) / 60
                    header_text += f"Audio file duration: {approx_minutes:.2f} minutes\n"
                except Exception:
                    header_text += f"Audio file duration: Estimated via pipeline tokens\n"

                header_text += "----------------------------------------\n\n"
                full_payload = header_text + transcript_body

                # Stream file writing directly up to Google Drive via Memory Stream
                print(f"   ⏳ Uploading finalized transcript to Drive...")
                text_stream = io.BytesIO(full_payload.encode('utf-8'))
                cloud_file_metadata = {'name': transcript_name, 'parents': [output_folder_id]}
                media_payload = MediaIoBaseUpload(text_stream, mimetype='text/plain', resumable=True)
                
                drive_service.files().create(body=cloud_file_metadata, media_body=media_payload).execute()

                total_file_time = time.time() - file_start_time
                print(f"   ✅ Saved to Drive: {transcript_name} (⏱️ Total time: {total_file_time/60:.2f}m)")

                success = True
                backoff_attempt = 0
                print("   ⏳ Sleeping 30 seconds to ease concurrent connections...")
                time.sleep(30)

            except APIError as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["429", "exhausted", "503", "unavailable", "504", "deadline"]):
                    backoff_attempt += 1
                    if backoff_attempt > MAX_RETRIES:
                        print(f"\n❌ CRITICAL: Reached maximum retry limit ({MAX_RETRIES}). Resource quota exhausted.")
                        return  
                    sleep_cooldown = 60 * backoff_attempt
                    print(f"\n⚠️ Rate limit/server pressure. Pausing for {sleep_cooldown}s before retry...")
                    if audio_upload:
                        try: gemini_client.files.delete(name=audio_upload.name)
                        except Exception: pass
                        audio_upload = None
                    time.sleep(sleep_cooldown)
                else:
                    print(f"   ❌ Terminal API exception: {e}")
                    break
            except Exception as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["timeout", "timed out", "read_timeout"]):
                    backoff_attempt += 1
                    if backoff_attempt > MAX_RETRIES:
                        print(f"\n❌ CRITICAL: Reached maximum retry limit ({MAX_RETRIES}) due to persistent timeouts.")
                        return
                    sleep_cooldown = 60 * backoff_attempt
                    print(f"\n⚠️ Connection timed out. Pausing for {sleep_cooldown}s before retry...")
                    if audio_upload:
                        try: gemini_client.files.delete(name=audio_upload.name)
                        except Exception: pass
                        audio_upload = None
                    time.sleep(sleep_cooldown)
                else:
                    print(f"   ❌ Unexpected local script failure: {e}")
                    break
            finally:
                if success and audio_upload:
                    try: gemini_client.files.delete(name=audio_upload.name)
                    except Exception: pass
                    audio_upload = None

        # Local scratch disk cleanup per loop item
        if os.path.exists(temp_local_audio):
            os.remove(temp_local_audio)

    print("\n🎉 Batch script complete! All transcripts checked and sync'd with Google Drive.")

# =============================================================================
# RUNTIME INVOCATION LANDING ZONE
# =============================================================================
if __name__ == "__main__":
    # ⚠️ TODO: Replace these strings with your actual target Input (MP3) and Output (TXT) Folder IDs from your Google Drive URL bar
    DRIVE_INPUT_FOLDER = "17MGWbLC8Qq_UxLCtOePc9aJgVSAakhde"
    DRIVE_OUTPUT_FOLDER = "17MGWbLC8Qq_UxLCtOePc9aJgVSAakhde"

    run_cloud_batch_transcription(
        input_folder_id=DRIVE_INPUT_FOLDER,
        output_folder_id=DRIVE_OUTPUT_FOLDER
    )
