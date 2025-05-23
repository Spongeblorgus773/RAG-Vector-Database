# D:\YouTubeTranscriptScraper\scripts\download_ia_collection.py
# (Complete version including --checksum, --log, AND --exclude_pattern arguments)
import os
import subprocess
import argparse
import sys
import traceback # Added for more detailed error info on unexpected issues
from urllib.parse import urlparse

# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# print(f"[*] Script directory: {SCRIPT_DIR}") # Optional: for debugging
# print(f"[*] Determined Project root: {PROJECT_ROOT_DIR}") # Optional: for debugging

# --- Default Configuration ---
DEFAULT_BASE_DOWNLOAD_DIR = os.path.join(PROJECT_ROOT_DIR, "data_sources", "internet_archive_downloads")
DEFAULT_FILE_PATTERN = "*.pdf" # Default to downloading PDFs
DEFAULT_EXCLUDE_PATTERN = "*_text.pdf" # Default pattern to exclude

def extract_collection_id_from_url(url):
    """Extracts the collection identifier from an archive.org details URL."""
    try:
        parsed = urlparse(url)
        if parsed.netloc == 'archive.org' and parsed.path.startswith('/details/'):
            identifier = parsed.path.split('/details/', 1)[1].strip('/')
            identifier = identifier.split('/')[0] # Handle potential extra path segments like /details/id/tab
            if identifier:
                return identifier
    except Exception:
        pass # Ignore parsing errors
    return None

def run_ia_download(collection_id, pattern, output_dir, no_subdirs, use_checksum, use_log, exclude_pattern):
    """Constructs and runs the ia download command."""
    print(f"[*] Preparing to download '{pattern}' files for collection: {collection_id}")
    if exclude_pattern: # Only print if there's an exclude pattern
        print(f"[*] Excluding files matching: {exclude_pattern}")
    print(f"[*] Target directory: {output_dir}")
    print(f"[*] Create item subdirectories: {not no_subdirs}")
    print(f"[*] Use Checksum: {use_checksum}")
    print(f"[*] Create Log File: {use_log}")

    # Ensure target directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Ensured output directory exists.")
    except OSError as e:
        print(f"[ERROR] Failed to create output directory '{output_dir}': {e}", file=sys.stderr)
        return False

    # Construct the command arguments
    command = [
        "ia", # The command-line tool executable
        "download",
        "--search", f"collection:{collection_id}", # Use search to target items in collection
        "--glob", pattern, # General pattern for files to include
    ]

    # Add exclude pattern if provided and not an empty string
    if exclude_pattern: 
        command.extend(["--exclude", exclude_pattern])

    command.extend(["--destdir", output_dir]) # Add destdir

    # Add other optional flags based on arguments
    if no_subdirs:
        command.append("--no-directories")
    if use_checksum:
        command.append("--checksum")
    if use_log:
        command.append("--log")

    print(f"[*] Executing command: {' '.join(command)}")

    try:
        # Run the command, check for errors, show output directly in terminal
        process = subprocess.run(command, check=True, text=True)
        print(f"[*] Command executed successfully for collection '{collection_id}'.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 'ia' command failed with return code {e.returncode} for collection '{collection_id}'.", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("[ERROR] 'ia' command not found. Is the internetarchive package installed and the venv active?", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc() 
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from an Internet Archive collection using the 'ia' command-line tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
        )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-id", "--identifier", help="The Internet Archive collection identifier (e.g., 'military-manuals').")
    input_group.add_argument("-url", "--collection_url", help="The full URL of the Internet Archive collection (e.g., 'https://archive.org/details/military-manuals').")

    parser.add_argument("-p", "--pattern", default=DEFAULT_FILE_PATTERN,
                        help=f"Glob pattern for files to download. Use quotes if pattern includes wildcards (*).")
    parser.add_argument("--exclude_pattern", default=DEFAULT_EXCLUDE_PATTERN,
                        help="Glob pattern for files to EXCLUDE from download (e.g., '*_text.pdf', '*.djvu'). Set to an empty string '--exclude_pattern \"\"' to disable exclusion.")
    parser.add_argument("-o", "--output_dir", default=None,
                        help=f"Destination directory for downloads. If omitted, creates '<project_root>/data_sources/internet_archive_downloads/<collection_id>'.")
    parser.add_argument("--no_subdirs", action='store_true',
                        help="Download all files directly into the output directory without item-specific subfolders (WARNING: may cause filename collisions).")
    parser.add_argument("--checksum", action='store_true',
                        help="Verify checksums and skip download if local file matches remote (for incremental downloads).")
    parser.add_argument("--log", action='store_true',
                        help="Generate a detailed log file (internetarchive.log) by the 'ia' tool, typically in the directory where the script is run.")

    args = parser.parse_args()

    collection_identifier = args.identifier
    if args.collection_url:
        extracted_id = extract_collection_id_from_url(args.collection_url)
        if not extracted_id:
            print(f"[ERROR] Could not extract a valid collection identifier from URL: {args.collection_url}", file=sys.stderr)
            sys.exit(1)
        collection_identifier = extracted_id
        print(f"[*] Extracted collection identifier '{collection_identifier}' from URL.")

    if not collection_identifier:
         print(f"[ERROR] Collection identifier could not be determined.", file=sys.stderr)
         sys.exit(1)

    output_directory = args.output_dir
    if not output_directory:
        output_directory = os.path.join(DEFAULT_BASE_DOWNLOAD_DIR, collection_identifier)
        print(f"[*] No output directory specified, using default: {output_directory}")
    
    effective_exclude_pattern = args.exclude_pattern if args.exclude_pattern else None

    success = run_ia_download(
        collection_identifier,
        args.pattern,
        output_directory,
        args.no_subdirs,
        args.checksum,
        args.log,
        effective_exclude_pattern 
    )

    if success:
        print("\n[*] Download process initiated successfully.")
    else:
        print("\n[*] Download process failed.")
        sys.exit(1)