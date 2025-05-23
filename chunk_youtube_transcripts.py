# D:\YouTubeTranscriptScraper\scripts\chunk_youtube_transcripts.py (FIXED)
import os
import json
import re
import traceback
import argparse # For optional command-line arguments
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    print("[Warning] Could not determine script directory automatically, using current working directory.")

# **MODIFIED: Project Root is one level up from the 'scripts' directory**
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)

print(f"[*] YouTube Chunker Script directory: {SCRIPT_DIR}")
print(f"[*] YouTube Chunker Determined Project root: {PROJECT_ROOT_DIR}")

# --- Configuration (using paths relative to the corrected PROJECT_ROOT_DIR) ---
# **MODIFIED: Paths updated to match target structure**
DEFAULT_INPUT_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, "data_sources", "youtube_transcripts")
DEFAULT_OUTPUT_JSONL_FILE = os.path.join(PROJECT_ROOT_DIR, "output", "knowledge_base_chunks.jsonl") # Shared output file
DEFAULT_PROCESSED_LOG_FILE = os.path.join(PROJECT_ROOT_DIR, "logs", "log_chunk_youtube.txt") # Correct log file path and name

CHUNK_SIZE = 1000 # From original script
CHUNK_OVERLAP = 150 # From original script

# Print the dynamically determined default paths for confirmation
print(f"[*] Default Input directory set to: {DEFAULT_INPUT_ROOT_DIR}")
print(f"[*] Default Output file set to: {DEFAULT_OUTPUT_JSONL_FILE}")
print(f"[*] Default Processed log file set to: {DEFAULT_PROCESSED_LOG_FILE}")

# --- LangChain Text Splitter Initialization ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=False, # Langchain default, keeps it simple
)

# --- Helper Functions ---
def load_processed_files(log_file):
    """Loads the set of already processed relative file paths."""
    processed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        # Store with forward slashes for consistency
                        processed.add(cleaned_line.replace("\\", "/"))
        except Exception as e:
            print(f"[Warning] Could not read processed log file {log_file}: {e}")
    return processed

def log_processed_file(filepath, log_file, project_root):
    """Appends the relative path of a processed file to the log."""
    try:
        relative_path = os.path.relpath(filepath, start=project_root)
        # Standardize path separators to forward slashes for consistency
        relative_path = relative_path.replace("\\", "/")

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir, exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(relative_path + '\n') # Write the relative path
    except ValueError as ve:
         print(f"    [Error] Could not calculate relative path for logging {os.path.basename(filepath)}: {ve}. File not logged.")
    except Exception as e:
        print(f"    [Error] Could not write to processed log file {log_file} for {os.path.basename(filepath)}: {e}")

# --- Main Processing Logic ---
# **MODIFIED: Pass project_root explicitly**
def process_transcripts(input_dir, output_jsonl, processed_log, project_root):
    processed_files_in_run = 0
    skipped_files_count = 0
    failed_files_count = 0
    new_chunks_count = 0

    print(f"\nStarting YouTube incremental chunking process...")
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output file (append mode): {os.path.abspath(output_jsonl)}")
    print(f"Processed files log (relative paths): {os.path.abspath(processed_log)}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Ensure output/log directories exist
    output_parent_dir = os.path.dirname(output_jsonl)
    log_parent_dir = os.path.dirname(processed_log)
    if output_parent_dir and not os.path.exists(output_parent_dir):
        try: os.makedirs(output_parent_dir, exist_ok=True)
        except OSError as e: print(f"[ERROR] Could not create output directory {output_parent_dir}: {e}. Cannot proceed."); return
    if log_parent_dir and not os.path.exists(log_parent_dir):
         try: os.makedirs(log_parent_dir, exist_ok=True)
         except OSError as e: print(f"[Warning] Could not create log directory {log_parent_dir}: {e}. Logging may fail.");

    already_processed_relative_paths = load_processed_files(processed_log)
    print(f"Found {len(already_processed_relative_paths)} file paths in the processed log.")

    try:
        # Open in append mode
        with open(output_jsonl, 'a', encoding='utf-8') as outfile:
            # Walk through the input directory
            for root, dirs, files in os.walk(input_dir):
                # Optional: Skip hidden directories if needed
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                print(f"\nProcessing directory: {root}")

                json_files_in_dir = [f for f in files if f.endswith(".json")]
                if not json_files_in_dir:
                    print("  No JSON files found in this directory.")
                    continue

                for filename in json_files_in_dir:
                    json_filepath = os.path.join(root, filename)
                    current_relative_path = None # Initialize

                    # --- Check if already processed USING RELATIVE PATH ---
                    try:
                        # Use the passed project_root for correct relative path calculation
                        current_relative_path = os.path.relpath(json_filepath, start=project_root).replace("\\", "/")
                        if current_relative_path in already_processed_relative_paths:
                            # print(f"    Skipping (already processed): {filename}")
                            skipped_files_count += 1
                            continue # Skip this file
                    except ValueError as ve:
                        # Handle cases where file might be outside the project root unexpectedly
                        print(f"    [Warning] Could not calculate relative path for checking {filename}: {ve}. Will attempt processing, but might duplicate if run again.")
                        current_relative_path = None # Ensure it's None if calculation failed
                    # -------------------------------------------------------

                    # --- Process the new file ---
                    print(f"  Processing new file: {filename}")
                    try:
                        with open(json_filepath, 'r', encoding='utf-8') as infile:
                            data = json.load(infile)

                        # Extract required fields
                        transcript = data.get("transcript")
                        video_id = data.get("video_id")

                        # Basic validation
                        if not transcript or not isinstance(transcript, str) or not transcript.strip():
                            print(f"    [SKIP] No valid transcript text found in {filename}.")
                            if current_relative_path: # Log only if relative path calculation succeeded
                                 log_processed_file(json_filepath, processed_log, project_root)
                                 already_processed_relative_paths.add(current_relative_path) # Add to set for current run
                            failed_files_count += 1
                            continue
                        if not video_id:
                             print(f"    [SKIP] Missing video_id in {filename}.")
                             if current_relative_path: # Log only if relative path calculation succeeded
                                 log_processed_file(json_filepath, processed_log, project_root)
                                 already_processed_relative_paths.add(current_relative_path) # Add to set for current run
                             failed_files_count += 1
                             continue

                        # Extract metadata
                        title = data.get("title", "Untitled Video")
                        channel_name_raw = data.get("channel_name_raw", "Unknown Channel")
                        channel_folder = data.get("channel_folder", "Unknown Channel Folder") # Added based on scrape script
                        video_url = data.get("video_url", f"https://www.youtube.com/watch?v={video_id}") # Added
                        upload_date = data.get("upload_date") # Already formatted in scrape script
                        # Add other metadata fields from the JSON if needed

                        # Chunk the transcript
                        cleaned_transcript = transcript.strip() # Remove leading/trailing whitespace
                        chunks = text_splitter.split_text(cleaned_transcript)
                        file_chunks_count = 0

                        # Process each chunk
                        for i, chunk_text in enumerate(chunks):
                            trimmed_chunk = chunk_text.strip()
                            if not trimmed_chunk: # Skip potentially empty chunks after stripping
                                continue

                            chunk_id = f"{video_id}_chunk_{i+1}"
                            chunk_metadata = {
                                "source_type": "youtube", # Added for consistency
                                "video_id": video_id,
                                "title": title,
                                "channel_name_raw": channel_name_raw,
                                "channel_folder": channel_folder,
                                "video_url": video_url,
                                "upload_date": upload_date,
                                "chunk_number": i + 1,
                                "total_chunks": len(chunks),
                                "source_json_relpath": current_relative_path, # Store relative path of source JSON
                                # Add any other relevant metadata here
                            }
                            # Remove None values from metadata for cleaner output
                            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}

                            chunk_data = {
                                "id": chunk_id,
                                "text": trimmed_chunk,
                                "metadata": chunk_metadata
                            }
                            # Write chunk as a JSON line
                            json.dump(chunk_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            new_chunks_count += 1
                            file_chunks_count += 1

                        print(f"    -> Created {file_chunks_count} chunks.")
                        # Log AFTER successful processing using relative path
                        if current_relative_path: # Log only if relative path calculation succeeded
                             log_processed_file(json_filepath, processed_log, project_root)
                             already_processed_relative_paths.add(current_relative_path) # Add to set for current run
                        processed_files_in_run += 1

                    except json.JSONDecodeError:
                        print(f"    [ERROR] Failed to decode JSON: {filename}")
                        failed_files_count += 1
                    except Exception as e:
                        print(f"    [ERROR] Unexpected error processing {filename}: {type(e).__name__} - {e}")
                        traceback.print_exc()
                        failed_files_count += 1
                        # Decide if you want to log errors as "processed" to avoid retrying constantly
                        # if current_relative_path:
                        #     log_processed_file(json_filepath, processed_log, project_root)
                        #     already_processed_relative_paths.add(current_relative_path)

    except IOError as e:
         print(f"[ERROR] Could not open output file {output_jsonl} for writing: {e}")
         # Optional: Re-raise or handle differently if needed

    # --- Final Summary ---
    print("\n--- YouTube Incremental Chunking Complete ---")
    print(f"New files processed and logged in this run: {processed_files_in_run}")
    print(f"Chunks added to {os.path.basename(output_jsonl)} in this run: {new_chunks_count}")
    print(f"Skipped previously processed files: {skipped_files_count}")
    print(f"Failed/Skipped files (JSON errors, missing data, other errors): {failed_files_count}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally chunk transcript files using relative paths for logging.")
    # Use the dynamically determined absolute paths as defaults for arguments
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_ROOT_DIR, help="Root directory containing transcript JSON files.")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_JSONL_FILE, help="SHARED JSONL file to append chunks to.")
    parser.add_argument("--log_file", default=DEFAULT_PROCESSED_LOG_FILE, help="File to log processed transcript relative paths.")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Target size for text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap size between chunks.")

    args = parser.parse_args()

    # Update text_splitter if chunk size/overlap are passed via CLI
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, add_start_index=False
    )

    # Pass PROJECT_ROOT_DIR explicitly to the main function
    process_transcripts(args.input_dir, args.output_file, args.log_file, PROJECT_ROOT_DIR)