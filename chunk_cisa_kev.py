# D:\YouTubeTranscriptScraper\scripts\chunk_cisa_kev.py
import os
import json
import traceback
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
# **MODIFIED: Project Root is one level up from the 'scripts' directory**
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"[*] CISA KEV Chunker Script directory: {SCRIPT_DIR}")
print(f"[*] CISA KEV Chunker Determined Project root: {PROJECT_ROOT_DIR}")

# --- Configuration (Using New Structure) ---
DEFAULT_CISA_KEV_JSON = os.path.join(PROJECT_ROOT_DIR, "data_sources", "cisa_kev", "known_exploited_vulnerabilities.json")
DEFAULT_SHARED_OUTPUT_JSONL = os.path.join(PROJECT_ROOT_DIR, "output", "knowledge_base_chunks.jsonl")
DEFAULT_PROCESSED_KEV_LOG = os.path.join(PROJECT_ROOT_DIR, "logs", "log_chunk_cisa_kev.txt") # Renamed log
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Print Configuration ---
print(f"[*] Default CISA KEV Input JSON set to: {DEFAULT_CISA_KEV_JSON}")
print(f"[*] Default Processed KEV Log file set to: {DEFAULT_PROCESSED_KEV_LOG}")
print(f"[*] Shared Output file (append mode) set to: {DEFAULT_SHARED_OUTPUT_JSONL}")

# --- LangChain Text Splitter Initialization ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=False,
)

# --- Helper Functions for Logging Processed CVE IDs ---
def load_processed_cve_ids(log_file):
    """Loads the set of already processed CVE IDs from the log file."""
    processed_ids = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    cve_id = line.strip()
                    if cve_id:
                        processed_ids.add(cve_id)
            print(f"[*] Loaded {len(processed_ids)} processed CVE IDs from {os.path.basename(log_file)}")
        except Exception as e:
            print(f"[Warning] Could not read processed CVE log file {log_file}: {e}")
    else:
        print(f"[*] Processed CVE log file not found ({os.path.basename(log_file)}). Assuming no CVEs processed yet.")
    return processed_ids

def log_processed_cve_id(cve_id, log_file):
    """Appends a successfully processed CVE ID to the log file."""
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(cve_id + '\n')
    except Exception as e:
        print(f"    [Error] Could not write CVE ID to log file {log_file} for {cve_id}: {e}")

# --- Main Processing Logic ---
# **MODIFIED: Pass project_root to the function**
def process_cisa_kev_catalog(input_json_path, output_jsonl_path, processed_log_path, project_root):
    processed_cves_in_run = 0
    skipped_cves_count = 0
    failed_cves_count = 0
    new_chunks_count = 0
    total_cves_in_file = 0

    print(f"\nStarting incremental CISA KEV processing...")
    print(f"Input KEV JSON: {os.path.abspath(input_json_path)}")
    print(f"Output file (append mode): {os.path.abspath(output_jsonl_path)}")
    print(f"Processed CVEs log: {os.path.abspath(processed_log_path)}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    if not os.path.isfile(input_json_path):
        print(f"[ERROR] Input CISA KEV JSON file not found: {input_json_path}")
        return

    output_parent_dir = os.path.dirname(output_jsonl_path)
    log_parent_dir = os.path.dirname(processed_log_path)
    if output_parent_dir and not os.path.exists(output_parent_dir):
        try: os.makedirs(output_parent_dir, exist_ok=True)
        except OSError as e: print(f"[ERROR] Could not create output directory {output_parent_dir}: {e}. Cannot proceed."); return
    if log_parent_dir and not os.path.exists(log_parent_dir):
         try: os.makedirs(log_parent_dir, exist_ok=True)
         except OSError as e: print(f"[Warning] Could not create log directory {log_parent_dir}: {e}. Logging may fail.");

    already_processed_cve_ids = load_processed_cve_ids(processed_log_path)

    try:
        with open(input_json_path, 'r', encoding='utf-8') as infile, \
             open(output_jsonl_path, 'a', encoding='utf-8') as outfile:

            data = json.load(infile)
            vulnerabilities = data.get('vulnerabilities', [])
            total_cves_in_file = len(vulnerabilities)
            print(f"[*] Found {total_cves_in_file} CVE entries in the input file.")

            if not vulnerabilities: print("[*] No vulnerabilities found in the JSON file."); return

            for vulnerability in vulnerabilities:
                cve_id = vulnerability.get("cveID")
                if not cve_id: print("    [Warning] Skipping entry with missing cveID."); failed_cves_count += 1; continue

                if cve_id in already_processed_cve_ids: skipped_cves_count += 1; continue

                print(f"  Processing new CVE: {cve_id}")
                try:
                    vuln_name = vulnerability.get("vulnerabilityName", "N/A")
                    short_desc = vulnerability.get("shortDescription", "N/A")
                    req_action = vulnerability.get("requiredAction", "N/A")
                    notes = vulnerability.get("notes", "")

                    combined_text = f"Vulnerability: {vuln_name}\nDescription: {short_desc}\nRequired Action: {req_action}"
                    if notes: combined_text += f"\nNotes: {notes}"

                    chunks = text_splitter.split_text(combined_text.strip())
                    file_chunks_count = 0

                    # Calculate relative path for metadata
                    input_file_rel_path = None
                    try:
                        abs_input_path = os.path.abspath(input_json_path)
                        input_file_rel_path = os.path.relpath(abs_input_path, start=project_root).replace("\\", "/")
                    except ValueError: input_file_rel_path = os.path.basename(input_json_path) # Fallback

                    base_metadata = {
                        "source_type": "cisa_kev", "source_file_relpath": input_file_rel_path,
                        "cveID": cve_id, "vulnerabilityName": vuln_name,
                        "vendorProject": vulnerability.get("vendorProject", "N/A"),
                        "product": vulnerability.get("product", "N/A"),
                        "dateAdded": vulnerability.get("dateAdded", "N/A"),
                        "dueDate": vulnerability.get("dueDate", "N/A"),
                        "knownRansomwareCampaignUse": vulnerability.get("knownRansomwareCampaignUse", "Unknown"),
                        "requiredAction": req_action, "shortDescription": short_desc,
                        "notes": notes if notes else None, "cwes": vulnerability.get("cwes")
                    }
                    base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

                    for i, chunk_text in enumerate(chunks):
                        trimmed_chunk = chunk_text.strip()
                        if not trimmed_chunk: continue

                        chunk_id = f"{cve_id}_chunk_{i+1}"
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata["chunk_number"] = i + 1
                        chunk_metadata["total_chunks"] = len(chunks)

                        chunk_data = {"id": chunk_id, "text": trimmed_chunk, "metadata": chunk_metadata}
                        json.dump(chunk_data, outfile, ensure_ascii=False); outfile.write('\n')
                        new_chunks_count += 1; file_chunks_count += 1

                    print(f"    -> Created {file_chunks_count} chunks.")
                    log_processed_cve_id(cve_id, processed_log_path)
                    already_processed_cve_ids.add(cve_id)
                    processed_cves_in_run += 1

                except Exception as e:
                    print(f"    [ERROR] Failed to process CVE {cve_id}: {type(e).__name__} - {e}")
                    traceback.print_exc(); failed_cves_count += 1

    except json.JSONDecodeError as json_err: print(f"[ERROR] Failed to decode input JSON file {input_json_path}: {json_err}")
    except IOError as e: print(f"[ERROR] Could not open or write file: {e}")
    except Exception as global_e: print(f"[FATAL ERROR] Unexpected error: {global_e}"); traceback.print_exc()

    print("\n--- CISA KEV Incremental Processing Complete ---")
    print(f"Total CVE entries in input file: {total_cves_in_file}")
    print(f"New CVEs processed and logged: {processed_cves_in_run}")
    print(f"Chunks added to output file: {new_chunks_count}")
    print(f"Skipped previously processed CVEs: {skipped_cves_count}")
    print(f"Failed CVE entries: {failed_cves_count}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally process CISA KEV JSON, chunk data, and append to a shared JSONL file.")
    parser.add_argument("--input_file", default=DEFAULT_CISA_KEV_JSON, help="Path to the input CISA KEV JSON file.")
    parser.add_argument("--output_file", default=DEFAULT_SHARED_OUTPUT_JSONL, help="SHARED JSONL file to append chunks to.")
    parser.add_argument("--log_file", default=DEFAULT_PROCESSED_KEV_LOG, help="File to log processed CISA KEV CVE IDs.")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Target size for text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help="Overlap size between chunks.")
    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, add_start_index=False
    )
    # **MODIFIED: Pass project_root to main function**
    process_cisa_kev_catalog(args.input_file, args.output_file, args.log_file, PROJECT_ROOT_DIR)