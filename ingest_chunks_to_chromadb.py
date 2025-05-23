# D:\YouTubeTranscriptScraper\scripts\ingest_chunks_to_chromadb.py
import os
import sys # Import sys
import json
import torch
import argparse
from tqdm import tqdm
import traceback
import time # <--- ADDED MISSING IMPORT

# Langchain Imports (ensure these are installed in venv)
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import filter_complex_metadata # Added for potential general use
except ImportError as e:
    print(f"[Error] Failed to import LangChain components: {e}")
    print("Please ensure 'langchain-chroma', 'langchain-huggingface', and 'langchain-community' are installed.")
    sys.exit(1)


# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"[*] Ingest Script directory: {SCRIPT_DIR}")
print(f"[*] Ingest Determined Project root: {PROJECT_ROOT_DIR}")

# --- Configuration (Using New Structure) ---
DEFAULT_JSONL_FILE = os.path.join(PROJECT_ROOT_DIR, "output", "knowledge_base_chunks.jsonl")
# This is your confirmed working SSD path
DEFAULT_DB_DIR = "C:\\YouTubeTranscriptScraper\\database\\chroma_db_multi_source"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Default batch size set to a moderate value
BATCH_SIZE = 5 # Your script had 5, let's keep it or you can adjust

# --- Print Configuration ---
print(f"[*] Input JSONL file set to: {DEFAULT_JSONL_FILE}")
print(f"[*] ChromaDB directory set to: {DEFAULT_DB_DIR}") # This will print your SSD path
print(f"[*] Embedding Model: {MODEL_NAME}")
# BATCH_SIZE will be printed later, after parsing args

# --- Determine Device for Embeddings ---
def get_device():
    if torch.cuda.is_available():
        try:
             _ = torch.tensor([1.0]).to('cuda')
             device = "cuda"
             print(f"[*] CUDA GPU found. Using device: {torch.cuda.get_device_name(0)}")
             return device
        except Exception as e: print(f"[Warning] CUDA found but failed to initialize: {e}. Falling back to CPU.")
    device = "cpu"
    print("[*] No compatible GPU detected or usable. Using device: CPU")
    return device

# --- Initialize Embedding Model ---
def initialize_embeddings(device):
    print(f"[*] Loading embedding model '{MODEL_NAME}' onto device '{device}'...")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            show_progress=False
        )
        print("[*] Performing test embedding...")
        _ = embedding_function.embed_query("test")
        print("[*] Embedding model loaded and tested successfully.")
        return embedding_function
    except Exception as e:
        print(f"[ERROR] initializing embedding model on device '{device}': {e}"); traceback.print_exc(); sys.exit(1)

# --- Initialize ChromaDB ---
def initialize_db(db_dir, embedding_func):
    print(f"[*] Initializing/Loading ChromaDB from: '{db_dir}'")
    os.makedirs(db_dir, exist_ok=True) # Ensures SSD path exists
    try:
        vector_db = Chroma(persist_directory=db_dir, embedding_function=embedding_func)
        try:
            count = vector_db._collection.count()
            print(f"[*] Connected to DB. Collection contains {count} documents.")
        except Exception as count_e:
            print(f"[Warning] Could not get initial document count from DB: {count_e}")
            print("[*] Connected to DB. Initial document count unknown (proceeding with caution).")
        return vector_db
    except Exception as e:
        print(f"[ERROR] initializing Chroma DB from '{db_dir}': {e}"); traceback.print_exc(); sys.exit(1)

# --- Get Existing IDs ---
def get_existing_ids(vector_db):
    print("[*] Fetching existing document IDs from database...")
    print("[INFO] This step can be very slow if the database is large or experiencing issues.")
    start_time_get_ids = time.time()
    try:
        existing_data = vector_db.get(include=[])
        existing_ids = set(existing_data['ids'])
        end_time_get_ids = time.time()
        print(f"[*] Found {len(existing_ids)} existing IDs in the database. (Time taken: {end_time_get_ids - start_time_get_ids:.2f} seconds)")
        return existing_ids
    except Exception as e:
        end_time_get_ids = time.time()
        print(f"[Error] fetching existing IDs from ChromaDB: {e} (Time taken: {end_time_get_ids - start_time_get_ids:.2f} seconds)"); traceback.print_exc()
        print("[Warning] Proceeding without ID check due to error. May add duplicate chunks if run multiple times after failures without this check.")
        return set()

# --- Main Ingestion Logic ---
def ingest_data(jsonl_file, vector_db, current_batch_size):
    print(f"[*] Reading chunks from: '{jsonl_file}'")
    all_texts = []
    all_metadatas = []
    all_ids = []
    line_count = 0
    skipped_malformed_lines = 0
    loaded_count = 0

    try:
        with open(jsonl_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line_count = line_num
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    chunk_id = data.get("id")
                    text = data.get("text")
                    metadata_from_jsonl = data.get("metadata")

                    if text and chunk_id and isinstance(metadata_from_jsonl, dict):
                        
                        # --- START OF METADATA FIX ---
                        processed_metadata = {}
                        for key, value in metadata_from_jsonl.items():
                            if isinstance(value, list):
                                if value:  # If the list is not empty
                                    # Convert each item in the list to string before joining
                                    processed_metadata[key] = ', '.join(map(str, value))
                                else: # If the list is empty
                                    processed_metadata[key] = 'N/A' 
                            elif value is None: # Handle None values explicitly
                                processed_metadata[key] = 'N/A' 
                            else:
                                processed_metadata[key] = value
                        
                        # Final check to ensure all values are basic types for ChromaDB
                        final_clean_metadata = {}
                        for k, v in processed_metadata.items():
                            if isinstance(v, (str, int, float, bool)):
                                final_clean_metadata[k] = v
                            else:
                                # Fallback for any other unexpected types: convert to string
                                print(f"  [Debug] Converting metadata value of type {type(v)} to string for key '{k}' in chunk_id '{chunk_id}'. Original value: {v}")
                                final_clean_metadata[k] = str(v)
                        # --- END OF METADATA FIX ---

                        all_texts.append(text)
                        all_metadatas.append(final_clean_metadata) # Use the cleaned metadata
                        all_ids.append(chunk_id)
                        loaded_count += 1
                    else:
                        missing = [k for k,v in [('id',chunk_id),('text',text),('metadata',metadata_from_jsonl)] if not v or (k=='metadata' and not isinstance(v,dict))]
                        print(f"  [Warning] Skipping line {line_count}. Missing/invalid: {', '.join(missing)}")
                        skipped_malformed_lines += 1
                except json.JSONDecodeError: print(f"  [Warning] Skipping line {line_count}: JSON decode error."); skipped_malformed_lines+=1
                except Exception as line_err: print(f"  [Warning] Skipping line {line_count}: Error processing line: {line_err}"); traceback.print_exc(); skipped_malformed_lines+=1 # Added traceback
    except FileNotFoundError: print(f"[ERROR] Input file not found: '{jsonl_file}'"); return
    except Exception as e: print(f"[ERROR] reading file '{jsonl_file}': {e}"); return

    print(f"[*] Total lines read: {line_count}")
    print(f"[*] Lines skipped due to malformation/errors: {skipped_malformed_lines}")
    print(f"[*] Valid chunks loaded from JSONL: {loaded_count}")
    if not all_ids: print("[*] No valid chunks loaded to process."); return

    existing_ids_set = get_existing_ids(vector_db)
    new_texts, new_metadatas_to_add, new_ids_to_add = [], [], [] # Corrected variable name
    skipped_existing_count = 0

    print("[*] Filtering chunks against existing IDs...")
    for i in tqdm(range(len(all_ids)), desc="Filtering Chunks", unit="chunk"):
        if all_ids[i] not in existing_ids_set:
            new_texts.append(all_texts[i])
            new_metadatas_to_add.append(all_metadatas[i]) # Use the correct variable
            new_ids_to_add.append(all_ids[i])
        else: skipped_existing_count += 1

    print(f"[*] Identified {len(new_ids_to_add)} new chunks to add.")
    print(f"[*] Skipped {skipped_existing_count} existing chunks.")
    if not new_ids_to_add: print("[*] No new chunks to add."); return

    total_added_successfully = 0
    print(f"[*] Ingesting {len(new_ids_to_add)} new chunks into ChromaDB (Batch size: {current_batch_size})...")

    if current_batch_size == 1: # Your script's logic for batch_size 1
        for i in tqdm(range(len(new_ids_to_add)), desc="Ingesting Individually", unit="document"):
            doc_text = [new_texts[i]]
            doc_metadata = [new_metadatas_to_add[i]] # Use the correct variable
            doc_id = [new_ids_to_add[i]]
            try:
                vector_db.add_texts(texts=doc_text, metadatas=doc_metadata, ids=doc_id)
                total_added_successfully += 1
            except Exception as e:
                print(f"\n--- [ERROR] adding document ID: {doc_id[0]} ---")
                print(f"    Problematic metadata: {doc_metadata[0]}") # Log the problematic metadata
                print(f"    Error: {type(e).__name__}: {e}")
                print(f"    Skipping this document.")
                if "Index with capacity 100" in str(e) and "cannot add 1 records" in str(e):
                    print("[FATAL] The 'Index capacity 100' error occurred even with BATCH_SIZE = 1.")
                    # ... (rest of your fatal error handling)
    else: # Original batching logic for BATCH_SIZE > 1
        num_batches = (len(new_ids_to_add) + current_batch_size - 1) // current_batch_size
        for i in tqdm(range(num_batches), desc="Ingesting Batches", unit="batch"):
            start_idx = i * current_batch_size
            end_idx = min(start_idx + current_batch_size, len(new_ids_to_add))
            batch_texts = new_texts[start_idx:end_idx]
            batch_metadatas = new_metadatas_to_add[start_idx:end_idx] # Use the correct variable
            batch_ids = new_ids_to_add[start_idx:end_idx]

            if not batch_ids: continue
            try:
                vector_db.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
                total_added_successfully += len(batch_ids)
            except Exception as e:
                print(f"\n--- [ERROR] adding batch {i+1}/{num_batches} (IDs: {batch_ids[0]}...) ---")
                if batch_metadatas:
                     print(f"    Sample problematic metadata from batch: {batch_metadatas[0]}")
                print(f"    Error: {type(e).__name__}: {e}")
                print(f"    Skipping this batch. Consider reducing batch size further if this error persists or inspect metadata.")

    print("\n--- Incremental Ingestion Complete ---")
    print(f"[*] Successfully added {total_added_successfully} new chunks to the database.")
    if total_added_successfully < len(new_ids_to_add): print(f"[*] Note: {len(new_ids_to_add) - total_added_successfully} chunks may have failed during processing.")

    try:
        final_count = vector_db._collection.count()
        print(f"[*] Total documents in collection now: {final_count}")
    except Exception as e: print(f"[Warning] Could not get final document count: {e}")
    print(f"[*] Database changes persisted in: {os.path.abspath(vector_db._persist_directory)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally ingest chunks from a JSONL file into ChromaDB.")
    parser.add_argument("--input_file", default=DEFAULT_JSONL_FILE, help=f"Path to the input JSONL file (default: {DEFAULT_JSONL_FILE}).")
    # The --db_dir argument allows overriding the DEFAULT_DB_DIR if needed for flexibility
    parser.add_argument("--db_dir", default=None, help=f"Path to the ChromaDB persistent directory (Optional, overrides default: {DEFAULT_DB_DIR}).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Number of documents to add per batch (default: {BATCH_SIZE}). For testing capacity errors, try 1.")
    args = parser.parse_args()

    # Determine the effective DB directory
    db_directory_to_use = args.db_dir if args.db_dir else DEFAULT_DB_DIR
    # Update the print statement to reflect the actual DB directory being used
    # This was missing in your original script if --db_dir was used from CLI
    if args.db_dir:
        print(f"[*] ChromaDB directory overridden by command line: {db_directory_to_use}")


    effective_batch_size = args.batch_size
    print(f"[*] Using Batch Size: {effective_batch_size}")

    selected_device = get_device()
    embed_func = initialize_embeddings(selected_device)
    # Pass the determined db_directory_to_use to initialize_db
    db = initialize_db(db_directory_to_use, embed_func)
    ingest_data(args.input_file, db, effective_batch_size)