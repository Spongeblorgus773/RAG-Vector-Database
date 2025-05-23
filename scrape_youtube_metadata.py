# D:\YouTubeTranscriptScraper\scripts\scrape_youtube_metadata.py
import os
import re
import yt_dlp
import sys
import json
import time
import random
import requests
import traceback
import argparse
from urllib.parse import urlparse
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, AgeRestricted, VideoUnplayable
from youtube_transcript_api._transcripts import FetchedTranscript, FetchedTranscriptSnippet # Kept for reference
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_IMPORTED = True
except ImportError:
    TQDM_IMPORTED = False

# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
except NameError:
    SCRIPT_DIR = os.getcwd()
    PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    print("[Warning] Could not determine script directory automatically.")

print(f"[*] Scraper Script directory: {SCRIPT_DIR}")
print(f"[*] Scraper Determined Project root: {PROJECT_ROOT_DIR}")

# === CONFIGURATION ===
DEFAULT_CHANNEL_URLS = [
      "https://www.youtube.com/@examplechannel/videos", "https://www.youtube.com/@examplechannel2/videos", "https://www.youtube.com/@examplechannel3/videos"
]
DEFAULT_ROOT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT_DIR, "data_sources", "youtube_transcripts")
DEFAULT_COMPLETED_LOG_FILE = os.path.join(PROJECT_ROOT_DIR, "logs", "log_scrape_youtube.txt")
DEFAULT_FFMPEG_PATH = os.path.join(PROJECT_ROOT_DIR, "ffmpeg", "bin", "ffmpeg.exe")
REQUEST_DELAY_SECONDS = 0.5
TRANSCRIPT_LANGUAGES = ['en', 'en-US']

STATUS_FETCHED = "FETCHED"; STATUS_NO_TRANSCRIPT_FOUND = "NO_TRANSCRIPT_FOUND"
STATUS_DISABLED = "DISABLED"; STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_AGE_RESTRICTED = "AGE_RESTRICTED"; STATUS_MEMBERS_ONLY = "MEMBERS_ONLY"
STATUS_NETWORK_ERROR = "NETWORK_ERROR"; STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"
BLOCKING_STATUSES = {STATUS_DISABLED, STATUS_UNAVAILABLE, STATUS_AGE_RESTRICTED, STATUS_MEMBERS_ONLY, STATUS_NO_TRANSCRIPT_FOUND}

print(f"[*] Output base directory set to: {DEFAULT_ROOT_OUTPUT_FOLDER}")
print(f"[*] Completed files log set to: {DEFAULT_COMPLETED_LOG_FILE}")
if os.path.exists(DEFAULT_FFMPEG_PATH): print(f"[*] FFmpeg path: {DEFAULT_FFMPEG_PATH}")
else: print(f"[Warning] FFmpeg path not found: {DEFAULT_FFMPEG_PATH}")

def sanitize_filename(title, max_length=240):
    leading_at = False
    if title and isinstance(title, str) and title.startswith('@'):
        leading_at = True; title = title[1:]
    if not title or not isinstance(title, str): title = "untitled_video"
    title = re.sub(r'[\\/*?:"<>|]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    if leading_at: title = '@' + title
    if not title or title == '@': title = "untitled_video"
    safe_max_length = max_length - 20
    if len(title) > safe_max_length:
        original_leading_at = title.startswith('@')
        if original_leading_at:
             title_to_truncate = title[1:]; truncated = title_to_truncate[:safe_max_length-1].strip(); title = '@' + truncated
        else: title = title[:safe_max_length].strip()
        while title and (title.endswith('.') or title.endswith(' ')): title = title[:-1]
    if not title or title == '@': title = f"video_invalid_title_{random.randint(1000,9999)}"
    return title

def get_channel_name(channel_url):
    try:
        match_at = re.search(r'youtube\.com/(@[^/?#]+)', channel_url)
        match_c = re.search(r'youtube\.com/c/([^/?#]+)', channel_url)
        match_user = re.search(r'youtube\.com/user/([^/?#]+)', channel_url)
        match_channel_id = re.search(r'youtube\.com/channel/([^/?#]+)', channel_url)
        potential_name = None
        if match_at: potential_name = match_at.group(1)
        elif match_c: potential_name = match_c.group(1)
        elif match_user: potential_name = match_user.group(1)
        elif match_channel_id: potential_name = match_channel_id.group(1)
        else:
            parsed_url = urlparse(channel_url); path = parsed_url.path.strip("/"); parts = path.split("/")
            if parts:
                last_part = parts[-1]; suffixes_to_remove = ["videos", "featured", "community", "about", "streams", "shorts", "playlists"]
                if last_part.lower() not in suffixes_to_remove and last_part: potential_name = last_part
                elif len(parts) > 1 and parts[-2]: potential_name = parts[-2]
                elif parts[0]: potential_name = parts[0]
        if potential_name: return sanitize_filename(potential_name, max_length=100)
        fallback_name = re.sub(r'\W+', '_', channel_url.replace("https://","").replace("http://",""))
        return sanitize_filename(fallback_name[:50], max_length=50)
    except Exception as e: print(f"Error parsing channel name from URL {channel_url}: {e}"); return "unknown_channel"

def get_video_list(channel_url, ffmpeg_path):
    ydl_opts = {
        'quiet': True, 'no_warnings': True, 'extract_flat': 'in_playlist',
        'force_generic_extractor': False, 'skip_download': True, 'ignoreerrors': True,
        'ffmpeg_location': ffmpeg_path if ffmpeg_path and os.path.exists(ffmpeg_path) else None,
        'cookiefile': os.path.join(PROJECT_ROOT_DIR, 'cookies.txt') if os.path.exists(os.path.join(PROJECT_ROOT_DIR, 'cookies.txt')) else None,
        'retries': 3, 'socket_timeout': 30,
    }
    ydl_opts = {k: v for k, v in ydl_opts.items() if v is not None}
    video_entries = []
    print(f"Attempting to fetch basic video list for: {channel_url}...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if info and isinstance(info.get('entries'), list):
                count = 0
                for entry in info['entries']:
                    if isinstance(entry, dict) and entry.get('id') and entry.get('title'): video_entries.append(entry); count += 1
                    elif entry: print(f"\nWarning: Skipping potentially incomplete/invalid video entry...")
                print(f"Extracted {count} valid video entries.")
            elif info and info.get('id') and info.get('title'): video_entries.append(info); print("Extracted 1 valid video entry (single video URL).")
            else: print(f"Warning: yt-dlp returned info, but no 'entries' list or valid single video found for {channel_url}.")
    except yt_dlp.utils.DownloadError as e: print(f"yt-dlp DownloadError fetching list for {channel_url}: {e}")
    except Exception as e: print(f"Unexpected error in get_video_list for {channel_url}: {type(e).__name__} - {e}"); traceback.print_exc()
    if not video_entries: print(f"Warning: Failed to retrieve any valid video entries for {channel_url}.")
    return video_entries

def get_transcript_text(video_id, languages=TRANSCRIPT_LANGUAGES):
    transcript_text = None; status = STATUS_UNKNOWN_ERROR; should_delay = True
    try:
        transcript_list_segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        lines = [segment['text'].strip() for segment in transcript_list_segments if isinstance(segment, dict) and 'text' in segment]
        if lines:
            transcript_text = " ".join(lines)
            if transcript_text.strip(): status = STATUS_FETCHED
            else: transcript_text = None; status = STATUS_NO_TRANSCRIPT_FOUND
        else: status = STATUS_NO_TRANSCRIPT_FOUND
    except TranscriptsDisabled: status = STATUS_DISABLED; print(f"    Transcripts disabled for {video_id}.")
    except NoTranscriptFound: status = STATUS_NO_TRANSCRIPT_FOUND; should_delay = False; print(f"    No transcript found by get_transcript() for languages {languages} for {video_id}.")
    except VideoUnavailable: status = STATUS_UNAVAILABLE; print(f"    Video unavailable for {video_id}.")
    except AgeRestricted: status = STATUS_AGE_RESTRICTED; print(f"    Video {video_id} is age restricted.")
    except VideoUnplayable as e: status = STATUS_MEMBERS_ONLY if 'members-only' in str(e).lower() else STATUS_UNAVAILABLE; print(f"    Video {video_id} unplayable: {status}")
    except requests.exceptions.RequestException as net_err: status = STATUS_NETWORK_ERROR; print(f"\n[NETWORK ERROR] fetching transcript for {video_id}: {net_err}")
    except Exception as e: status = STATUS_UNKNOWN_ERROR; print(f"\n[ERROR] Transcript API/processing error for {video_id}: {type(e).__name__} - {e}"); traceback.print_exc()
    if should_delay and REQUEST_DELAY_SECONDS > 0:
        delay_duration = max(0.1, REQUEST_DELAY_SECONDS + (random.random()*REQUEST_DELAY_SECONDS*0.2) - (REQUEST_DELAY_SECONDS*0.1)); time.sleep(delay_duration)
    return transcript_text, status

def save_json(data, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except OSError as e: print(f"\n[FILE OS ERROR] writing {os.path.basename(filepath)}: {e}"); return False
    except Exception as e: print(f"\n[FILE WRITE ERROR] writing {os.path.basename(filepath)}: {type(e).__name__} - {e}"); return False

def load_completed_files(log_file):
    completed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f: path = line.strip(); completed.add(path.replace("\\", "/")) if path else None
        except Exception as e: print(f"[Warning] Could not read completed log file {log_file}: {e}")
    else: print(f"Completed log file not found ({os.path.basename(log_file)}). Starting fresh.")
    return completed

def log_completed_file(filepath, log_file, project_root):
    try:
        abs_filepath = os.path.abspath(filepath)
        relative_path = os.path.relpath(abs_filepath, start=project_root).replace("\\", "/")
        log_dir = os.path.dirname(log_file); os.makedirs(log_dir, exist_ok=True) if log_dir and not os.path.exists(log_dir) else None
        with open(log_file, 'a', encoding='utf-8') as f: f.write(relative_path + '\n')
    except ValueError as ve: print(f"    [Error] Could not calculate relative path for logging {os.path.basename(filepath)}: {ve}. File not logged.")
    except Exception as e: print(f"\n[Error] Could not write to completed log file {log_file} for {os.path.basename(filepath)}: {e}")

def main(channel_urls, output_base_dir, log_file, ffmpeg_path):
    start_time = time.time(); print("="*50 + "\nStarting YouTube Metadata & Transcript Scraper\n" + "="*50)
    print(f"Target languages: {TRANSCRIPT_LANGUAGES}, Delay: ~{REQUEST_DELAY_SECONDS:.2f}s (+/- 10% jitter)")
    print(f"Output base directory: {os.path.abspath(output_base_dir)}")
    print(f"Completed files log (relative paths): {os.path.abspath(log_file)}")
    if ffmpeg_path and os.path.exists(ffmpeg_path): print(f"FFmpeg path: {ffmpeg_path}")
    elif ffmpeg_path: print(f"WARNING: FFmpeg path specified but not found: {ffmpeg_path}")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    completed_files_set = load_completed_files(log_file)
    print(f"Loaded {len(completed_files_set)} paths from completed log.")

    files_to_retry = []
    total_processed_overall, total_new_files_saved_with_transcript, total_new_files_saved_metadata_only = 0, 0, 0
    total_checked_existing_files, total_skipped_existing_with_transcript, total_skipped_due_to_status_pass1 = 0, 0, 0
    total_skipped_completed_log, total_retries_deferred = 0, 0
    total_retried_missing_transcript, total_retry_successful, total_retry_failed = 0, 0, 0
    total_file_read_errors, total_file_write_errors, total_channel_errors, total_api_errors = 0, 0, 0, 0

    print("\n--- PASS 1: Scanning Channels & Processing Videos ---")
    for channel_url in channel_urls:
        channel_start_time = time.time(); print(f"\n>>> Processing Channel URL: {channel_url} <<<")
        channel_name_raw = get_channel_name(channel_url)
        if channel_name_raw == "unknown_channel": print(f"Skipping URL {channel_url} due to channel name parsing error."); total_channel_errors += 1; continue
        channel_folder_name = sanitize_filename(channel_name_raw, max_length=100)
        channel_output_dir = os.path.join(output_base_dir, channel_folder_name)
        try: os.makedirs(channel_output_dir, exist_ok=True)
        except OSError as e: print(f"Error creating directory {channel_output_dir}: {e}. Skipping channel."); total_channel_errors += 1; continue

        video_metadata_list = get_video_list(channel_url, ffmpeg_path)
        if not video_metadata_list: print(f"No videos found or error fetching for '{channel_folder_name}'. Skipping channel."); total_channel_errors += 1; continue
        num_videos_in_channel = len(video_metadata_list)
        print(f"Found {num_videos_in_channel} videos listed for '{channel_folder_name}'. Checking status...")

        channel_processed,ch_skipped_completed,ch_skipped_exist_transcript,ch_skipped_status_p1,ch_new_transcript,ch_new_meta,ch_deferred,ch_read_err,ch_write_err,ch_checked_exist,ch_api_err = (0,)*11

        video_iterator = tqdm(enumerate(video_metadata_list,1),total=num_videos_in_channel,desc=f"'{channel_folder_name}'",unit="video",leave=True) if TQDM_IMPORTED else enumerate(video_metadata_list,1)
        if not TQDM_IMPORTED: print(f"Processing videos for '{channel_folder_name}'...")

        for i, video_info in video_iterator:
            channel_processed += 1; vid_id = video_info.get('id'); vid_title = video_info.get('title', 'Untitled Video')
            if not vid_id: (tqdm.write if TQDM_IMPORTED else print)(f"\nWarning: Skipping entry {i} missing Video ID."); continue
            upload_date_str = video_info.get('upload_date'); formatted_date, date_suffix = (None, "")
            if upload_date_str:
                try: date_obj=datetime.strptime(upload_date_str,'%Y%m%d');formatted_date=date_obj.strftime('%Y-%m-%d');date_suffix=formatted_date
                except(ValueError,TypeError):formatted_date=None;date_suffix=upload_date_str if isinstance(upload_date_str,str)else""
            clean_title=sanitize_filename(vid_title);filename_base=f"{clean_title}{' ('+date_suffix+')'if date_suffix else''}"
            max_base_len=240-(len(vid_id)+7);filename_base=filename_base[:max_base_len].strip() if len(filename_base)>max_base_len else filename_base
            while filename_base.endswith('.')or filename_base.endswith(' '):filename_base=filename_base[:-1]
            if not filename_base:filename_base="video"
            json_filename=f"{filename_base}_({vid_id}).json";json_filepath=os.path.join(channel_output_dir,json_filename)
            current_relative_path=None
            try:current_relative_path=os.path.relpath(json_filepath,start=PROJECT_ROOT_DIR).replace("\\","/")
            except ValueError as ve:(tqdm.write if TQDM_IMPORTED else print)(f"\n[Warning] Relpath calc error for {json_filename}:{ve}")

            if current_relative_path and current_relative_path in completed_files_set:
                total_skipped_completed_log+=1;ch_skipped_completed+=1;continue

            is_complete_for_log_pass1 = False
            if os.path.exists(json_filepath):
                total_checked_existing_files+=1;ch_checked_exist+=1
                try:
                    with open(json_filepath,"r",encoding="utf-8")as f:existing_data=json.load(f)
                    if existing_data.get('transcript'):
                        total_skipped_existing_with_transcript+=1;ch_skipped_exist_transcript+=1;is_complete_for_log_pass1=True
                    else: # No transcript, ALWAYS add to retry for Pass 2
                        if(json_filepath,vid_id,vid_title)not in files_to_retry:files_to_retry.append((json_filepath,vid_id,vid_title))
                        total_retries_deferred+=1;ch_deferred+=1;is_complete_for_log_pass1=False # Not complete for Pass1 log
                except Exception as e:
                    (tqdm.write if TQDM_IMPORTED else print)(f"\n[Error/Corrupt] Reading {json_filename}:{e}. Retrying.")
                    if(json_filepath,vid_id,vid_title)not in files_to_retry:files_to_retry.append((json_filepath,vid_id,vid_title))
                    total_retries_deferred+=1;ch_deferred+=1;total_file_read_errors+=1;ch_read_err+=1;is_complete_for_log_pass1=False
            else:
                total_processed_overall+=1
                transcript_text,status_string=get_transcript_text(vid_id,languages=TRANSCRIPT_LANGUAGES)
                if status_string not in[STATUS_FETCHED,STATUS_NO_TRANSCRIPT_FOUND,STATUS_UNKNOWN_ERROR]:total_api_errors+=1;ch_api_err+=1
                video_data={"video_id":vid_id,"title":vid_title,"channel_name_raw":channel_name_raw,"channel_folder":channel_folder_name,
                            "video_url":video_info.get('webpage_url',f"https://www.youtube.com/watch?v={vid_id}"),"upload_date":formatted_date,
                            "original_upload_date_str":upload_date_str,"description":video_info.get('description'),"tags":video_info.get('tags'),
                            "categories":video_info.get('categories'),"view_count":video_info.get('view_count'),"like_count":video_info.get('like_count'),
                            "comment_count":video_info.get('comment_count'),"duration_seconds":video_info.get('duration'),
                            "duration_string":video_info.get('duration_string'),"thumbnail_url":video_info.get('thumbnail'),
                            "channel_id":video_info.get('channel_id'),"transcript":transcript_text,"status":status_string,
                            "transcript_language_preference":TRANSCRIPT_LANGUAGES,"timestamp_fetched_utc":datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
                if save_json(video_data,json_filepath):
                    if transcript_text is not None and status_string==STATUS_FETCHED:total_new_files_saved_with_transcript+=1;ch_new_transcript+=1;is_complete_for_log_pass1=True
                    else:total_new_files_saved_metadata_only+=1;ch_new_meta+=1
                    if status_string in BLOCKING_STATUSES:is_complete_for_log_pass1=True # e.g. NO_TRANSCRIPT_FOUND for a new file
                else:total_file_write_errors+=1;ch_write_err+=1;is_complete_for_log_pass1=False
            
            if is_complete_for_log_pass1 and current_relative_path and current_relative_path not in completed_files_set:
                log_completed_file(json_filepath,log_file,PROJECT_ROOT_DIR);completed_files_set.add(current_relative_path)

        if TQDM_IMPORTED: video_iterator.close()
        channel_duration = time.time()-channel_start_time
        print(f"\n--- Channel Scan Summary ('{channel_folder_name}') ---")
        print(f"Duration: {channel_duration:.2f}s | Videos Checked: {channel_processed}")
        print(f"  Skipped (in log): {ch_skipped_completed} | Skipped (existing w/ transcript): {ch_skipped_exist_transcript}")
        # total_skipped_due_to_status_pass1 is no longer relevant here as all non-transcript files go to retry.
        print(f"  New Saved (w/ transcript): {ch_new_transcript} | New Saved (metadata only): {ch_new_meta}")
        print(f"  Deferred for Retry (Pass 2): {ch_deferred} | Read/Write Errors: {ch_read_err+ch_write_err} | API Errors: {ch_api_err}")

    print(f"\n--- PASS 2: Processing {len(files_to_retry)} Deferred Retries ---")
    retry_start_time = time.time()
    if files_to_retry:
        retry_iterator = tqdm(files_to_retry,desc="Processing Retries",unit="video",leave=False) if TQDM_IMPORTED else files_to_retry
        if not TQDM_IMPORTED: print("Processing deferred retries...")
        for filepath,vid_id,vid_title in retry_iterator:
            if TQDM_IMPORTED:retry_iterator.set_postfix_str(f"Retrying {vid_id}",refresh=True)
            print(f"  [Pass 2] Attempting retry for {vid_id} ({vid_title})...")
            total_retried_missing_transcript+=1
            transcript_text,status_string=get_transcript_text(vid_id,languages=TRANSCRIPT_LANGUAGES)
            try:
                with open(filepath,"r",encoding="utf-8")as f:existing_data=json.load(f)
                existing_data["transcript"]=transcript_text;existing_data["status"]=status_string
                existing_data["timestamp_fetched_utc"]=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                if save_json(existing_data,filepath):
                    print(f"    -> [Pass 2] Updated {os.path.basename(filepath)} with status: {status_string}")
                    is_complete_retry=False
                    if status_string==STATUS_FETCHED:total_retry_successful+=1;is_complete_retry=True
                    elif status_string in BLOCKING_STATUSES:total_retry_failed+=1;is_complete_retry=True
                    else:total_retry_failed+=1
                    if is_complete_retry:
                        current_relative_path_retry=None
                        try:
                            current_relative_path_retry=os.path.relpath(filepath,start=PROJECT_ROOT_DIR).replace("\\","/")
                            if current_relative_path_retry not in completed_files_set:
                                log_completed_file(filepath,log_file,PROJECT_ROOT_DIR);completed_files_set.add(current_relative_path_retry)
                        except ValueError:print(f"    [Warning] Could not get relative path for logging retry of {vid_id}")
                else:print(f"    [ERROR] Failed to save updated JSON during retry for {vid_id}");total_retry_failed+=1;total_file_write_errors+=1
            except Exception as e:print(f"    [ERROR] Failed to read/process existing JSON for {vid_id} during retry: {e}");total_retry_failed+=1;total_file_read_errors+=1
        if TQDM_IMPORTED and retry_iterator:retry_iterator.close()
    else:print("[Info] No files were flagged for retry in Pass 1.")
    retry_duration=time.time()-retry_start_time
    print(f"\n--- Deferred Retry Phase Complete ---")
    print(f"Duration (Pass 2): {retry_duration:.2f}s | Retries attempted: {total_retried_missing_transcript}")
    print(f"  Successfully fetched transcript: {total_retry_successful} | Still missing/Error: {total_retry_failed}")

    end_time=time.time();total_duration=end_time-start_time
    print(f"\n{'='*20} All Processing Finished {'='*20}")
    print(f"Total script duration: {total_duration:.2f} seconds.")
    print(f"Overall files processed (new): {total_processed_overall}")
    print(f"  New files with transcript: {total_new_files_saved_with_transcript} | New files metadata only: {total_new_files_saved_metadata_only}")
    print(f"Existing files checked: {total_checked_existing_files}")
    print(f"  Skipped (already had transcript): {total_skipped_existing_with_transcript} | Skipped (in completed log): {total_skipped_completed_log}")
    print(f"Retries from existing files (Pass 2):")
    print(f"  Total deferred to Pass 2: {total_retries_deferred} | Retries attempted: {total_retried_missing_transcript}")
    print(f"  Successfully fetched transcript: {total_retry_successful} | Still missing transcript / Error: {total_retry_failed}")
    print(f"Errors encountered: Channel: {total_channel_errors} | API: {total_api_errors} | File Read: {total_file_read_errors} | File Write: {total_file_write_errors}")
    print(f"Final size of completed log: {len(completed_files_set)}")
    print("="*50 + "\nScript Finished.\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape YouTube channel videos, fetch transcripts, and save metadata.")
    parser.add_argument("--output_dir", default=DEFAULT_ROOT_OUTPUT_FOLDER, help="Base directory to save transcript JSON files.")
    parser.add_argument("--log_file", default=DEFAULT_COMPLETED_LOG_FILE, help="File to log completed relative transcript paths.")
    parser.add_argument("--ffmpeg_path", default=DEFAULT_FFMPEG_PATH, help="Path to the ffmpeg executable (optional).")
    parser.add_argument("--channels", nargs='+', help="List of channel URLs to process (overrides internal list).")
    args = parser.parse_args()
    channels_to_process = args.channels if args.channels else DEFAULT_CHANNEL_URLS
    if not channels_to_process: print("Error: No channel URLs specified."); sys.exit(1)
    main(channels_to_process, args.output_dir, args.log_file, args.ffmpeg_path)