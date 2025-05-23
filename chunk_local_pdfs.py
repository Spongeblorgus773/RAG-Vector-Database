# D:\YouTubeTranscriptScraper\scripts\chunk_local_pdfs.py (FIXED)
import os
import json
import traceback
import argparse
import fitz # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    print("[Warning] Could not determine script directory automatically, using current working directory.")

# **MODIFIED: Project Root is one level up from the 'scripts' directory**
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)

print(f"[*] PDF Chunker Script directory: {SCRIPT_DIR}")
print(f"[*] PDF Chunker Determined Project root: {PROJECT_ROOT_DIR}")

# --- Configuration ---
# **MODIFIED: Paths updated to match target structure**
DEFAULT_PDF_INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data_sources", "local_pdfs")
DEFAULT_PDF_PROCESSED_LOG = os.path.join(PROJECT_ROOT_DIR, "logs", "log_chunk_pdfs.txt") # Correct log file path and name
DEFAULT_SHARED_OUTPUT_JSONL = os.path.join(PROJECT_ROOT_DIR, "output", "knowledge_base_chunks.jsonl") # Shared output file

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Print Configuration ---
print(f"[*] Default PDF Input directory set to: {DEFAULT_PDF_INPUT_DIR}")
print(f"[*] Default PDF Processed log file set to: {DEFAULT_PDF_PROCESSED_LOG}")
print(f"[*] Shared Output file (append mode) set to: {DEFAULT_SHARED_OUTPUT_JSONL}")

# --- LangChain Text Splitter Initialization ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=False,
)

# --- Helper Functions (Copied and verified) ---
def load_processed_files(log_file):
    """Loads the set of already processed relative file paths."""
    processed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        processed.add(cleaned_line.replace("\\", "/"))
        except Exception as e:
            print(f"[Warning] Could not read processed log file {log_file}: {e}")
    return processed

def log_processed_file(filepath, log_file, project_root):
    """Appends the relative path of a processed file to the log."""
    try:
        relative_path = os.path.relpath(filepath, start=project_root)
        relative_path = relative_path.replace("\\", "/")

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir, exist_ok=True)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(relative_path + '\n')
    except ValueError as ve:
         print(f"    [Error] Could not calculate relative path for logging {os.path.basename(filepath)}: {ve}. File not logged.")
    except Exception as e:
        print(f"    [Error] Could not write to processed log file {log_file} for {os.path.basename(filepath)}: {e}")

# --- Main PDF Processing Logic ---
# **MODIFIED: Pass project_root explicitly**
def process_pdf_directory(input_dir, output_jsonl, processed_log, project_root):
    processed_files_in_run = 0
    skipped_files_count = 0
    failed_files_count = 0
    new_chunks_count = 0

    print(f"\nStarting incremental PDF chunking process (using PyMuPDF/fitz)...")
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output file (append mode): {os.path.abspath(output_jsonl)}")
    print(f"Processed PDFs log (relative paths): {os.path.abspath(processed_log)}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] PDF Input directory not found: {input_dir}")
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
    print(f"Found {len(already_processed_relative_paths)} PDF paths in the processed log.")

    try:
        with open(output_jsonl, 'a', encoding='utf-8') as outfile:
            for root, dirs, files in os.walk(input_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')] # Skip hidden dirs
                print(f"\nProcessing directory: {root}")

                pdf_files_in_dir = [f for f in files if f.lower().endswith(".pdf")]
                if not pdf_files_in_dir:
                    print("  No PDF files found in this directory.")
                    continue

                for filename in pdf_files_in_dir:
                    pdf_filepath = os.path.join(root, filename)
                    current_relative_path = None

                    # --- Check if already processed ---
                    try:
                        current_relative_path = os.path.relpath(pdf_filepath, start=project_root).replace("\\", "/")
                        if current_relative_path in already_processed_relative_paths:
                            # print(f"    Skipping (already processed): {filename}")
                            skipped_files_count += 1
                            continue
                    except ValueError as ve:
                        print(f"    [Warning] Could not calculate relative path for checking {filename}: {ve}. Will attempt processing, but might duplicate if run again.")
                        current_relative_path = None

                    # --- Process the new PDF file using fitz ---
                    print(f"  Processing PDF: {filename}")
                    extracted_text = ""
                    pdf_metadata_from_doc = {} # Store optional metadata from fitz
                    try:
                        doc = fitz.open(pdf_filepath)
                        page_texts = []
                        # Try getting optional PDF metadata from fitz
                        try:
                            meta = doc.metadata
                            if meta:
                                pdf_metadata_from_doc['pdf_title'] = meta.get('title', '').strip() or None # Use None if empty
                                pdf_metadata_from_doc['pdf_author'] = meta.get('author', '').strip() or None
                                pdf_metadata_from_doc['pdf_subject'] = meta.get('subject', '').strip() or None
                                pdf_metadata_from_doc['pdf_keywords'] = meta.get('keywords', '').strip() or None
                                # Add other fitz metadata fields if needed
                        except Exception as meta_err:
                            print(f"    [Warning] Could not read metadata for {filename} using fitz: {meta_err}")

                        # Extract text from pages
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            page_text = page.get_text("text", sort=True) # Extract text, sorting helps reading order
                            if page_text:
                                page_texts.append(page_text.strip())
                        doc.close() # Close the fitz document
                        extracted_text = "\n\n".join(page_texts).strip() # Join pages with double newline

                        if not extracted_text:
                            print(f"    [SKIP] No text could be extracted from {filename} using fitz.")
                            if current_relative_path:
                                log_processed_file(pdf_filepath, processed_log, project_root)
                                already_processed_relative_paths.add(current_relative_path)
                            failed_files_count += 1
                            continue

                        # --- Chunking ---
                        chunks = text_splitter.split_text(extracted_text)
                        file_chunks_count = 0

                        # --- Format and Write Chunks ---
                        for i, chunk_text in enumerate(chunks):
                            trimmed_chunk = chunk_text.strip()
                            if not trimmed_chunk: continue

                            # Create a unique ID, using relative path if possible
                            chunk_id_base = current_relative_path if current_relative_path else os.path.basename(pdf_filepath)
                            chunk_id = f"{chunk_id_base}_chunk_{i+1}"

                            # Base metadata for PDF chunks
                            chunk_metadata = {
                                "source_type": "pdf",
                                "source_path_relative": current_relative_path, # Store relative path
                                "filename": filename,
                                "chunk_number": i + 1,
                                "total_chunks": len(chunks),
                            }
                            # Add metadata extracted from the PDF document itself
                            chunk_metadata.update(pdf_metadata_from_doc)
                            # Remove None values
                            chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}


                            chunk_data = {
                                "id": chunk_id,
                                "text": trimmed_chunk,
                                "metadata": chunk_metadata
                            }

                            json.dump(chunk_data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            new_chunks_count += 1
                            file_chunks_count += 1

                        print(f"    -> Created {file_chunks_count} chunks.")
                        if current_relative_path:
                             log_processed_file(pdf_filepath, processed_log, project_root)
                             already_processed_relative_paths.add(current_relative_path)
                        processed_files_in_run += 1

                    # Catch errors specific to file operations or fitz
                    except FileNotFoundError:
                         print(f"    [ERROR] File not found during processing attempt: {filename}")
                         failed_files_count += 1
                    except fitz.fitz.FileDataError as fitz_err: # More specific fitz error
                         print(f"    [ERROR] Fitz error processing {filename} (possibly corrupted or password-protected): {fitz_err}")
                         failed_files_count += 1
                    except Exception as e:
                        # Catch any other unexpected errors during processing of this file
                        print(f"    [ERROR] Unexpected error processing {filename} with fitz: {type(e).__name__} - {e}")
                        traceback.print_exc()
                        failed_files_count += 1
                        # Optional: Log error file as processed?
                        # if current_relative_path:
                        #     log_processed_file(pdf_filepath, processed_log, project_root)
                        #     already_processed_relative_paths.add(current_relative_path)


    except IOError as e:
         print(f"[ERROR] Could not open output file {output_jsonl} for writing: {e}")

    # --- Final Summary ---
    print("\n--- PDF Incremental Chunking Complete (using PyMuPDF/fitz) ---")
    print(f"New PDF files processed and logged in this run: {processed_files_in_run}")
    print(f"Chunks added to {os.path.basename(output_jsonl)} in this run: {new_chunks_count}")
    print(f"Skipped previously processed PDF files: {skipped_files_count}")
    print(f"Failed/Skipped PDF files (read errors, no text, fitz errors, other): {failed_files_count}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally chunk PDF files using PyMuPDF/fitz and append to a shared JSONL file.")
    parser.add_argument("--input_dir", default=DEFAULT_PDF_INPUT_DIR, help="Directory containing PDF files to process.")
    parser.add_argument("--output_file", default=DEFAULT_SHARED_OUTPUT_JSONL, help="SHARED JSONL file to append chunks to.")
    parser.add_argument("--log_file", default=DEFAULT_PDF_PROCESSED_LOG, help="File to log processed PDF relative paths.")
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
    process_pdf_directory(args.input_dir, args.output_file, args.log_file, PROJECT_ROOT_DIR)