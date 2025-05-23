# D:\YouTubeTranscriptScraper\scripts\NEw_query_rag_database.py
# GOAL:
# IF RERANKING: Reliably get scores from the underlying reranker model,
#               show score stats, then interactively prompt for threshold for the current query.
#               Filter by this interactive threshold.
# IF NOT RERANKING: Base retriever fetches a max K, then script filters by similarity score threshold.

import os
import sys
import json
import torch
import traceback
import argparse
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Optional, Tuple
import requests
import numpy as np # Import numpy for checking array type

# Langchain Imports
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    # ContextualCompressionRetriever is not strictly needed for the primary reranking path now,
    # but keeping it in case we want to compare or for other compressors later.
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
    from langchain_core.documents import Document
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    print(f"[Error] Failed to import core LangChain components: {e}")
    sys.exit(1)

# Re-ranking Imports
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    RERANKING_AVAILABLE = True
except ImportError:
    print("[Warning] Re-ranking components not found. Re-ranking will be disabled.")
    CrossEncoderReranker = None
    HuggingFaceCrossEncoder = None
    RERANKING_AVAILABLE = False

# LLM Import
try:
    from langchain_community.llms import Ollama as OllamaLLM
except ImportError:
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
         print("[Error] Failed to import Ollama LLM. Install 'langchain-community' or 'langchain-ollama'.")
         sys.exit(1)

# --- Script Paths & Basic Config ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"[*] Query Script directory: {SCRIPT_DIR}")
print(f"[*] Query Determined Project root: {PROJECT_ROOT_DIR}")

DEFAULT_DB_DIR = "C:\\YouTubeTranscriptScraper\\database\\chroma_db_multi_source"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "output")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

DEFAULT_RERANKER_ENABLED = RERANKING_AVAILABLE
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER = 50
DEFAULT_K_FOR_DIRECT_RETRIEVAL = 20

DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER = 0.0
DEFAULT_BASE_SIMILARITY_THRESHOLD = 0.75

AVAILABLE_MODELS = sorted(list(set(
    ["granite3.3:2b", "phi4-mini-reasoning", "dolphin3:8b","llama3.1:8b",
    "qwen3:4b", "gemma3:4b"] +
    ["llama3:latest", "mistral:latest", "codellama:latest", "phi3:latest",
    "qwen:latest", "gemma:latest"]
)))
MEMORY_RAG_PROMPT_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer the current 'QUESTION' based ONLY on the provided 'CONTEXT' (which may contain info from YouTube, CISA, PDFs etc.) and the 'CHAT HISTORY'.
- Synthesize information from the CONTEXT and CHAT HISTORY to provide a comprehensive answer to the QUESTION.
- If the answer isn't found in the CONTEXT or CHAT HISTORY, state that you cannot answer based on the provided information.
- Do not use any external knowledge.
- Do not output your reasoning or thinking process (e.g., avoid <think> tags). Output only the final answer directly.

CHAT HISTORY:
{history}

CONTEXT:
{context}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
QUESTION: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
OUTPUT_FILE_PREFIX = "rag_chat_session"
DEFAULT_MAX_TOKENS = 1024
EXIT_KEYWORDS = {"end", "stop", "quit", "bye", "exit"}
COMMAND_KEYWORDS = {"/set_tokens", "/help", "/mode", "/showchunks", "/set_rerank_threshold", "/set_base_threshold"}
DEFAULT_SESSION_MODE = "rag"
DEFAULT_SHOW_CHUNKS = True
# --- End Basic Config ---

# --- Helper Functions ---
def get_device():
    if torch.cuda.is_available():
        try: _ = torch.tensor([1.0]).to('cuda'); return "cuda"
        except Exception as e: print(f"[Warning] CUDA init failed: {e}. CPU fallback."); return "cpu"
    return "cpu"

def initialize_embeddings(device_to_use):
    print(f"[*] Embeddings: '{EMBEDDING_MODEL_NAME}' on '{device_to_use}'...")
    try:
        embed_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device_to_use}, encode_kwargs={'normalize_embeddings': True}, show_progress=False)
        _ = embed_func.embed_query("test"); print("[*] Embeddings loaded."); return embed_func
    except Exception as e: print(f"[ERROR] Embeddings init: {e}"); traceback.print_exc(); sys.exit(1)

def initialize_db(db_dir, embed_func):
    print(f"[*] ChromaDB from: '{db_dir}'")
    if not os.path.isdir(db_dir): print(f"[Error] DB dir not found: {db_dir}"); sys.exit(1)
    try:
        vector_db = Chroma(persist_directory=db_dir, embedding_function=embed_func)
        print(f"[*] DB Connected. Docs: {vector_db._collection.count()}")
        return vector_db
    except Exception as e: print(f"[ERROR] DB init: {e}"); traceback.print_exc(); sys.exit(1)

def format_docs_for_llm(docs: List[Document]) -> str:
    if not docs: return "No relevant documents found."
    formatted = []
    for i, doc in enumerate(docs):
        content = doc.page_content; metadata = doc.metadata
        score = metadata.get('final_score_used_for_filtering')
        score_info = f" (Score: {score:.4f})" if score is not None else ""
        title = metadata.get('title') or metadata.get('vulnerabilityName') or metadata.get('pdf_title') or metadata.get('filename', 'N/A')
        source_type = metadata.get('source_type', 'Unknown')
        chunk_num = metadata.get('chunk_number', '?')
        identifier = f" (CVE: {metadata.get('cveID')})" if metadata.get('cveID') else f" (Video: {metadata.get('video_id')})" if metadata.get('video_id') else ""
        formatted.append(f"--- Context Chunk {i+1} (Source: {source_type}{identifier}, Title: '{title}', Chunk: {chunk_num}){score_info} ---\n{content}")
    return "\n\n".join(formatted)

def check_ollama_connection(model_name=""):
    try:
        import ollama
        client = ollama.Client()
        models_info = client.list().get('models', [])
        if not isinstance(models_info, list):
            print("[Warning] Unexpected response format from Ollama list. Proceeding cautiously.")
            return True
        if model_name:
            if not any(m.get('name','').startswith(model_name.split(':')[0]) for m in models_info):
                print(f"[Error] Ollama model '{model_name}' not found. Pull it first, e.g., 'ollama pull {model_name}'."); return False
        return True
    except ImportError: print("[Error] 'ollama' Python library not found. Install with 'pip install ollama'."); return False
    except requests.exceptions.ConnectionError: print("[Error] Ollama server connection failed. Ensure Ollama is running."); return False
    except Exception as e: print(f"[Error] Ollama check encountered an issue: {type(e).__name__} - {e}"); return False

def display_retrieved_docs(query, docs, info_str):
    print(f"\n--- {len(docs)} Retrieved Docs for '{query}' ({info_str}) ---")
    if not docs: print("--- No documents. ---"); return
    for i, doc in enumerate(docs):
        metadata = doc.metadata; content = doc.page_content
        # Try to get any available score, prioritizing the one used for filtering
        score = metadata.get('final_score_used_for_filtering', metadata.get('relevance_score', metadata.get('similarity_score')))
        score_info = f"Score: {score:.4f} | " if score is not None else ""
        title = metadata.get('title') or metadata.get('vulnerabilityName') or metadata.get('pdf_title') or metadata.get('filename', 'N/A')
        source_type = metadata.get('source_type', 'Unknown')
        chunk_num = metadata.get('chunk_number', '?')
        doc_id_display = metadata.get('id_for_log') or \
                         metadata.get('video_id') or \
                         metadata.get('cveID') or \
                         metadata.get('id', f'doc_idx_{i}')

        identifier = f" (ID: {doc_id_display})" # Simplified identifier for display
        print(f"[{i+1}] {score_info}Source: {source_type}{identifier} | Title: '{title}' | Chunk: {chunk_num}")
        print(f"    Snippet: {content[:250].replace(chr(10), ' ')}...")
    print("-" * 80)

def initialize_llm(model_name, num_tokens):
    print(f"\n[*] LLM: '{model_name}', max_tokens={num_tokens}...")
    try:
        llm = OllamaLLM(model=model_name, num_predict=num_tokens)
        llm.invoke("Test: 1+1 is?"); print("[*] LLM initialized."); return llm
    except Exception as e: print(f"[ERROR] LLM init: {e}"); return None
# --- END Helper Functions ---

# --- Centralized Document Fetching and Filtering Function ---
def get_final_documents_for_turn(
    query_text: str,
    db: Chroma,
    is_reranking_active_now: bool,
    k_to_fetch_initially: int,
    reranker_obj: Optional[CrossEncoderReranker],
    base_score_thresh: float, # Session default for base similarity
    rerank_score_thresh: float # Session default for reranker, used as default for interactive prompt
    ) -> Tuple[List[Document], str]:

    final_documents: List[Document] = []
    info_string = ""
    docs_with_scores_to_filter: List[Document] = []

    if is_reranking_active_now and reranker_obj and reranker_obj.model:
        base_retriever = db.as_retriever(search_kwargs={"k": k_to_fetch_initially})
        try:
            initial_docs_from_db = base_retriever.invoke(query_text) # Updated to .invoke()
        except Exception as e_invoke:
            print(f"DEBUG: Error during base_retriever.invoke(): {e_invoke}")
            traceback.print_exc()
            initial_docs_from_db = []

        count_from_initial_retrieval = len(initial_docs_from_db)
        print(f"\nDEBUG: Retrieved {count_from_initial_retrieval} initial documents from DB using retriever.invoke().")

        raw_scores_list = [] # Will store float scores or None

        if initial_docs_from_db:
            print(f"DEBUG: Preparing to call reranker_obj.model.score() directly.")
            sentence_pairs = [(query_text, doc.page_content) for doc in initial_docs_from_db]
            
            try:
                if sentence_pairs:
                    print(f"DEBUG: Calling reranker_obj.model.score() with {len(sentence_pairs)} pairs.")
                    raw_scores_from_model = reranker_obj.model.score(sentence_pairs)
                    # print(f"DEBUG: Raw scores from reranker_obj.model.score() (first 10): {raw_scores_from_model[:10]}")
                    
                    if raw_scores_from_model is not None:
                        processed_scores_temp = []
                        if isinstance(raw_scores_from_model, np.ndarray):
                            processed_scores_temp = raw_scores_from_model.tolist()
                        elif isinstance(raw_scores_from_model, list):
                            processed_scores_temp = raw_scores_from_model
                        else:
                            print(f"DEBUG: raw_scores_from_model is of unexpected type: {type(raw_scores_from_model)}. Attempting to process.")
                            try: processed_scores_temp = list(raw_scores_from_model)
                            except TypeError: processed_scores_temp = []

                        if len(processed_scores_temp) == len(initial_docs_from_db):
                            for s_idx, s_val in enumerate(processed_scores_temp):
                                try: raw_scores_list.append(float(s_val))
                                except (ValueError, TypeError):
                                    # print(f"DEBUG: Could not convert score {s_val} at index {s_idx} to float. Setting to None.")
                                    raw_scores_list.append(None) # Store None if conversion fails
                        else:
                            print(f"DEBUG: Mismatch in length between scores ({len(processed_scores_temp)}) and documents ({len(initial_docs_from_db)}). No scores will be used from this batch.")
                    else:
                        print("DEBUG: raw_scores_from_model is None.")
                else:
                    print("DEBUG: No sentence pairs to score.")
            except Exception as e_score:
                print(f"DEBUG: Error during reranker_obj.model.score(): {e_score}")
                traceback.print_exc()
            
            # Attach scores to documents
            if raw_scores_list and len(raw_scores_list) == len(initial_docs_from_db):
                print("DEBUG: Attaching manually obtained raw_scores to documents.")
                for i, doc_original in enumerate(initial_docs_from_db):
                    # Create a new Document object to avoid modifying cached objects from retriever
                    new_doc = Document(page_content=doc_original.page_content, metadata=doc_original.metadata.copy())
                    new_doc.metadata['relevance_score'] = raw_scores_list[i]
                    # Try to get a persistent ID for logging
                    new_doc.metadata['id_for_log'] = new_doc.metadata.get('id') or \
                                                   new_doc.metadata.get('video_id') or \
                                                   new_doc.metadata.get('cveID') or \
                                                   f'orig_idx_{i}'
                    docs_with_scores_to_filter.append(new_doc)
            else:
                print("DEBUG: No usable scores obtained from direct model call. Reranking scores will not be applied for filtering.")
                # Fallback: use initial documents, but they won't have 'relevance_score' for this path
                docs_with_scores_to_filter = [Document(page_content=d.page_content, metadata=d.metadata.copy()) for d in initial_docs_from_db]
        else: 
            print("DEBUG: No initial documents from DB to rerank.")

        # --- NEW: Interactive Threshold Input ---
        actual_threshold_for_this_query = rerank_score_thresh # Default to session threshold

        valid_scores_for_stats = [doc.metadata['relevance_score'] for doc in docs_with_scores_to_filter if doc.metadata.get('relevance_score') is not None]

        if valid_scores_for_stats:
            min_s, max_s, avg_s = min(valid_scores_for_stats), max(valid_scores_for_stats), sum(valid_scores_for_stats)/len(valid_scores_for_stats)
            print(f"\n--- Score Stats for Current Query ---")
            print(f"Scores for {len(valid_scores_for_stats)} docs range from: {min_s:.4f} to {max_s:.4f} (Avg: {avg_s:.4f})")
            print(f"Highest score observed: {max_s:.4f}")
            
            while True:
                try:
                    threshold_input_str = input(f"Enter threshold for this query (session default: {rerank_score_thresh:.2f}, press Enter to use default): ").strip()
                    if not threshold_input_str:
                        actual_threshold_for_this_query = rerank_score_thresh
                        print(f"Using session default threshold: {actual_threshold_for_this_query:.2f}")
                    else:
                        actual_threshold_for_this_query = float(threshold_input_str)
                        print(f"Using threshold for this query: {actual_threshold_for_this_query:.2f}")
                    break 
                except ValueError:
                    print("Invalid input. Please enter a number or press Enter for default.")
        elif docs_with_scores_to_filter: 
             print("\nWARNING: No valid 'relevance_score' found in document metadata. Cannot apply score-based thresholding for this query using reranker scores.")
        else: 
            print("\nINFO: No documents were retrieved or scored to apply a threshold.")

        # Filter documents using the determined threshold for this query
        for doc_idx, doc in enumerate(docs_with_scores_to_filter):
            score = doc.metadata.get('relevance_score') 
            doc_id_for_log = doc.metadata.get('id_for_log', f'processed_idx_{doc_idx}')

            if score is not None and score >= actual_threshold_for_this_query:
                doc.metadata['final_score_used_for_filtering'] = score
                final_documents.append(doc)
            elif score is None and is_reranking_active_now and any(d.metadata.get('relevance_score') is not None for d in docs_with_scores_to_filter): # Only log if some docs had scores
                 print(f"  [Filter-Info] Doc ID '{doc_id_for_log}' had a None score or was not processed for scoring. Metadata: {doc.metadata}. Skipping.")
        
        info_string = (f"Reranked (Interactive Threshold): Initial DB k={k_to_fetch_initially} ({count_from_initial_retrieval} actual), "
                       f"{len(docs_with_scores_to_filter)} docs considered, "
                       f"Thresholded (score >= {actual_threshold_for_this_query:.2f}) to {len(final_documents)}")
    else: 
        docs_with_sim_scores = db.similarity_search_with_relevance_scores(query_text, k=k_to_fetch_initially)
        count_from_base_retrieval = len(docs_with_sim_scores)
        for doc, score in docs_with_sim_scores:
            if score >= base_score_thresh:
                doc.metadata['similarity_score'] = score
                doc.metadata['final_score_used_for_filtering'] = score
                final_documents.append(doc)
        info_string = (f"Direct: Max k={k_to_fetch_initially}, "
                       f"{count_from_base_retrieval} retrieved with scores, "
                       f"Thresholded (score >= {base_score_thresh:.2f}) to {len(final_documents)}")
    
    return final_documents, info_string

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chat Application")
    parser.add_argument("--db_dir", default=DEFAULT_DB_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    DB_DIR = args.db_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not check_ollama_connection(): sys.exit(1)
    embed_device = get_device()
    embedding_function = initialize_embeddings(embed_device)
    vector_db: Chroma = initialize_db(DB_DIR, embedding_function)

    session_mode = DEFAULT_SESSION_MODE
    show_chunks_in_terminal = DEFAULT_SHOW_CHUNKS
    selected_model_name = None; llm = None; max_tokens_value = DEFAULT_MAX_TOKENS

    # These will store the SESSION DEFAULTS for thresholds
    param_reranker_score_threshold_session_default = DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER
    param_base_similarity_threshold_session_default = DEFAULT_BASE_SIMILARITY_THRESHOLD
    
    # param_k_initial_retrieval will be set based on reranker status
    param_k_initial_retrieval = DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER if DEFAULT_RERANKER_ENABLED else DEFAULT_K_FOR_DIRECT_RETRIEVAL


    _choice = input(f"Mode (1. RAG, 2. Retrieval Only, Enter='{DEFAULT_SESSION_MODE}'): ").strip()
    session_mode = {"1": "rag", "2": "retrieval_only"}.get(_choice, DEFAULT_SESSION_MODE)
    print(f"[*] Mode: {session_mode.replace('_',' ').title()}")

    _choice = input(f"Show chunks in terminal? (Y/n, Enter={'Y' if DEFAULT_SHOW_CHUNKS else 'N'}): ").strip().lower()
    show_chunks_in_terminal = not (_choice == 'n')
    print(f"[*] Show chunks: {'ON' if show_chunks_in_terminal else 'OFF'}")

    if session_mode == "rag":
        print("\n--- Select LLM Model ---")
        default_llm_display = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else 'None (no models configured)'
        for i, name in enumerate(AVAILABLE_MODELS): print(f"  {i+1}. {name}")
        while True:
            try:
                _choice = input(f"Choose LLM (1-{len(AVAILABLE_MODELS)} or type name, Enter for '{default_llm_display}'): ").strip()
                temp_name = ""
                if not _choice and AVAILABLE_MODELS:
                    temp_name = AVAILABLE_MODELS[0] 
                    print(f"Defaulting to {temp_name}")
                elif _choice.isdigit():
                    model_index = int(_choice)-1
                    if 0 <= model_index < len(AVAILABLE_MODELS):
                        temp_name = AVAILABLE_MODELS[model_index]
                    else: print("Invalid choice number."); continue
                elif _choice: temp_name = _choice
                else: 
                    if not AVAILABLE_MODELS: print("No models configured. Cannot proceed in RAG mode without an LLM."); sys.exit(1)
                    print("No input. Please select a model or type a name."); continue

                if temp_name and check_ollama_connection(temp_name):
                    selected_model_name = temp_name; break 
                elif not temp_name and not AVAILABLE_MODELS: 
                     print("No models available to default to and no input given.")
                     continue
            except (ValueError, IndexError): print("Invalid selection. Please try again.")

        print(f"[*] Using LLM: {selected_model_name}")
        while True:
            try:
                _input_str = input(f"Max LLM response tokens? (Enter for default={DEFAULT_MAX_TOKENS}): ").strip()
                if not _input_str: max_tokens_value = DEFAULT_MAX_TOKENS; break
                max_tokens_value = int(_input_str)
                if max_tokens_value > 50: break
                else: print("Please enter a value greater than 50.")
            except ValueError: print("Invalid input. Please enter a number.")
        llm = initialize_llm(selected_model_name, max_tokens_value)
        if not llm: print("[Warning] LLM initialization failed, RAG mode may be impaired.")

    print("\n--- Configure Retrieval ---")
    if RERANKING_AVAILABLE:
        _choice = input(f"Enable Re-ranking? (Y/n, Enter={'Y' if DEFAULT_RERANKER_ENABLED else 'N'}): ").strip().lower()
        reranker_active_for_session = not (_choice == 'n')
    else:
        print("[Info] Re-ranking components unavailable. Re-ranking is DISABLED.")
        reranker_active_for_session = False

    reranker_model_instance = None 
    if reranker_active_for_session:
        print("Re-ranking is ON.")
        try:
            _input_str = input(f"Initial candidates for DB fetch (k)? (Default={DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER}): ").strip()
            param_k_initial_retrieval = int(_input_str) if _input_str else DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER
            if param_k_initial_retrieval <= 0: param_k_initial_retrieval = DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER
        except ValueError: param_k_initial_retrieval = DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER

        try:
            # This sets the SESSION DEFAULT threshold
            _input_str = input(f"Re-ranker score threshold (SESSION DEFAULT, e.g., -5.0 to 10.0. Default={DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER}): ").strip()
            param_reranker_score_threshold_session_default = float(_input_str) if _input_str else DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER
        except ValueError: param_reranker_score_threshold_session_default = DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER
        print(f"[*] Reranker: Will fetch {param_k_initial_retrieval} initial candidates from DB. SESSION DEFAULT threshold for scores >= {param_reranker_score_threshold_session_default:.2f}")
        
        if RERANKING_AVAILABLE and CrossEncoderReranker and HuggingFaceCrossEncoder: 
            print(f"[*] Initializing Re-ranker model: '{DEFAULT_RERANKER_MODEL_NAME}'...")
            try:
                encoder = HuggingFaceCrossEncoder(model_name=DEFAULT_RERANKER_MODEL_NAME, model_kwargs={'device': embed_device})
                # top_n for CrossEncoderReranker is how many docs it passes through if used as a compressor.
                # Since we call model.score() directly, this top_n isn't directly used in our primary path,
                # but good to set it consistently.
                reranker_model_instance = CrossEncoderReranker(model=encoder, top_n=param_k_initial_retrieval) 
                print(f"[*] Re-ranker model instance created (its 'top_n' is set to {param_k_initial_retrieval}).")
            except Exception as e:
                print(f"[Error] Re-ranker init failed: {e}"); traceback.print_exc()
                reranker_active_for_session = False 
                reranker_model_instance = None
        else: 
            print("[Error] Reranking was enabled, but necessary components are not available. Disabling reranking.")
            reranker_active_for_session = False
            reranker_model_instance = None
    else: 
        print("Re-ranking is OFF. Direct retrieval from vector store.")
        param_k_initial_retrieval = DEFAULT_K_FOR_DIRECT_RETRIEVAL # Set k for direct retrieval
        try:
            _input_str = input(f"Max documents to retrieve (before similarity threshold, Default={DEFAULT_K_FOR_DIRECT_RETRIEVAL}): ").strip()
            param_k_initial_retrieval = int(_input_str) if _input_str else DEFAULT_K_FOR_DIRECT_RETRIEVAL
            if param_k_initial_retrieval <=0: param_k_initial_retrieval = DEFAULT_K_FOR_DIRECT_RETRIEVAL
        except ValueError: param_k_initial_retrieval = DEFAULT_K_FOR_DIRECT_RETRIEVAL

        try:
            _input_str = input(f"Similarity score threshold (SESSION DEFAULT, 0-1, higher is better, Default={DEFAULT_BASE_SIMILARITY_THRESHOLD}): ").strip()
            param_base_similarity_threshold_session_default = float(_input_str) if _input_str else DEFAULT_BASE_SIMILARITY_THRESHOLD
            if not (0.0 <= param_base_similarity_threshold_session_default <= 1.0): param_base_similarity_threshold_session_default = DEFAULT_BASE_SIMILARITY_THRESHOLD
        except ValueError: param_base_similarity_threshold_session_default = DEFAULT_BASE_SIMILARITY_THRESHOLD
        print(f"[*] Direct Retrieval: Max k={param_k_initial_retrieval}, SESSION DEFAULT Threshold >={param_base_similarity_threshold_session_default:.2f}")
            
    memory = ConversationBufferMemory(memory_key="history", input_key="question", output_key="ai_response", return_messages=False)
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=MEMORY_RAG_PROMPT_TEMPLATE)
    follow_up_prompt = PromptTemplate(input_variables=["history", "context", "question"], template=MEMORY_RAG_PROMPT_TEMPLATE)
    rag_chain = None
    follow_up_chain = None

    if session_mode == "rag" and llm:
        def _docs_for_rag_chain_lambda(query_text: str) -> List[Document]:
            # Note: get_final_documents_for_turn will now interactively ask for threshold if reranking
            docs, _ = get_final_documents_for_turn(
                query_text, vector_db, 
                reranker_active_for_session and reranker_model_instance is not None, 
                param_k_initial_retrieval, 
                reranker_model_instance, 
                param_base_similarity_threshold_session_default, # Pass session default
                param_reranker_score_threshold_session_default    # Pass session default
            )
            return docs

        rag_chain = (
            RunnablePassthrough.assign(retrieved_docs=itemgetter("question") | RunnableLambda(_docs_for_rag_chain_lambda))
            | RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs", []))),
                history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question", "")}).get("history",""))
            ) | prompt | llm | StrOutputParser()
        )
        print("[*] RAG chain created.")
        follow_up_chain = ( 
             RunnablePassthrough.assign(history=RunnableLambda(lambda x: memory.load_memory_variables({"question":x.get("question","")}).get("history","")))
             | follow_up_prompt | llm | StrOutputParser() 
        )
        print("[*] Follow-up chain created.")

    print("\n" + "="*30 + " Chat Session Started " + "="*30)
    print(f"Session Mode: {session_mode.replace('_', ' ').title()}")
    if session_mode == "rag": print(f"LLM Model: {selected_model_name} | Max Tokens: {max_tokens_value}")
    
    if reranker_active_for_session and reranker_model_instance:
        print(f"Re-ranking: ENABLED (Initial k for DB fetch={param_k_initial_retrieval}, SESSION DEFAULT Score Threshold >={param_reranker_score_threshold_session_default:.2f})")
        print("INFO: For each query, you'll be shown score stats and prompted for a per-query threshold.")
    else:
        print(f"Re-ranking: DISABLED (Direct Retrieve Max k from DB={param_k_initial_retrieval}, SESSION DEFAULT Similarity Threshold >={param_base_similarity_threshold_session_default:.2f})")

    print(f"Show Chunks in Terminal: {'ENABLED' if show_chunks_in_terminal else 'DISABLED'}")
    print(f"Type '/help' for commands, '{', '.join(EXIT_KEYWORDS)}' to end.")
    print("="* (62 + len(" Chat Session Started ") -2)) 

    conversation_log = []
    turn_count = 0
    last_final_docs_for_follow_up: Optional[List[Document]] = None

    try:
        while True:
            turn_count += 1; is_follow_up = False
            if turn_count > 1 and session_mode == "rag" and last_final_docs_for_follow_up and llm:
                _choice = input("Follow-up on last context? (y/n, Enter=n): ").strip().lower()
                if _choice == 'y': is_follow_up = True; print("[Using previous context]")

            user_input = input(f"\n[{turn_count}] You: ").strip()
            if not user_input: turn_count -= 1; continue 
            if user_input.lower() in EXIT_KEYWORDS: print("\nExiting..."); break

            if user_input.startswith('/'):
                parts = user_input.split(); cmd = parts[0].lower(); 
                arg1_str = parts[1] if len(parts) > 1 else None
                
                if cmd == "/help":
                     print("\nAvailable commands:")
                     print("  /mode              : Toggle RAG/Retrieval Only.")
                     print("  /showchunks        : Toggle display of retrieved chunk details.")
                     print("  /set_tokens <N>    : Change AI response token limit (RAG mode only).")
                     print("  /set_rerank_threshold <float>: Change SESSION DEFAULT score threshold for re-ranked docs.")
                     print("                           (Note: You'll be prompted per query if reranking is on).")
                     print("  /set_base_threshold <0.0-1.0>: Change SESSION DEFAULT similarity threshold for direct retrieval.")
                     print(f"  {', '.join(EXIT_KEYWORDS)} : Exit.")
                elif cmd == "/set_rerank_threshold": # Sets the SESSION DEFAULT
                    if RERANKING_AVAILABLE : # Check if reranking could even be active
                        try:
                            if arg1_str is None: raise ValueError("Missing threshold value.")
                            param_reranker_score_threshold_session_default = float(arg1_str)
                            print(f"SESSION DEFAULT rerank threshold set to {param_reranker_score_threshold_session_default:.2f}")
                            if not (reranker_active_for_session and reranker_model_instance):
                                print("Note: Re-ranking is not currently active. This default will apply if it's enabled.")
                        except ValueError as e: print(f"Usage: /set_rerank_threshold <float_value>. Error: {e}")
                    else: print("Re-ranking components are not available in this environment.")
                elif cmd == "/set_base_threshold": # Sets the SESSION DEFAULT
                    try:
                        if arg1_str is None: raise ValueError("Missing threshold value.")
                        val = float(arg1_str)
                        if 0.0 <= val <= 1.0: 
                            param_base_similarity_threshold_session_default = val
                            print(f"SESSION DEFAULT base similarity threshold set to {param_base_similarity_threshold_session_default:.2f}")
                            if reranker_active_for_session and reranker_model_instance:
                                print("Note: Direct retrieval threshold is not used when re-ranking is active.")
                        else: print("Base threshold must be between 0.0 and 1.0.")
                    except ValueError as e: print(f"Usage: /set_base_threshold <float_value_0_to_1>. Error: {e}")
                elif cmd == "/mode":
                     session_mode = "retrieval_only" if session_mode == "rag" else "rag"
                     print(f"Switched to {session_mode.replace('_', ' ').title()} mode.")
                     if session_mode == "rag" and not llm and selected_model_name: 
                         llm = initialize_llm(selected_model_name, max_tokens_value)
                         if not llm: print("[Warning] LLM re-init failed.")
                         if llm: 
                            def _docs_for_rag_chain_lambda_rebuild(query_text: str) -> List[Document]:
                                docs, _ = get_final_documents_for_turn(query_text, vector_db, reranker_active_for_session and reranker_model_instance is not None, param_k_initial_retrieval, reranker_model_instance, param_base_similarity_threshold_session_default, param_reranker_score_threshold_session_default)
                                return docs
                            rag_chain = ( RunnablePassthrough.assign(retrieved_docs=itemgetter("question") | RunnableLambda(_docs_for_rag_chain_lambda_rebuild)) | RunnablePassthrough.assign( context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs", []))), history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question", "")}).get("history",""))) | prompt | llm | StrOutputParser())
                            follow_up_chain = ( RunnablePassthrough.assign(history=RunnableLambda(lambda x: memory.load_memory_variables({"question":x.get("question","")}).get("history",""))) | follow_up_prompt | llm | StrOutputParser())
                            print("[*] RAG chain (re-)created.")
                elif cmd == "/showchunks":
                     show_chunks_in_terminal = not show_chunks_in_terminal
                     print(f"Chunk display in terminal: {'ENABLED' if show_chunks_in_terminal else 'DISABLED'}")
                elif cmd == "/set_tokens":
                     if session_mode != "rag": print("Token limit only applicable in RAG mode.")
                     elif not llm or not selected_model_name : print("LLM not initialized.")
                     else:
                         print(f"Current max tokens: {max_tokens_value}")
                         try:
                             new_limit_str = arg1_str if arg1_str else input("Enter new max token limit (> 50): ").strip()
                             if not new_limit_str: raise ValueError("No token limit provided.")
                             new_limit = int(new_limit_str)
                             if new_limit > 50:
                                 new_llm_instance = initialize_llm(selected_model_name, new_limit) 
                                 if new_llm_instance: 
                                     llm = new_llm_instance 
                                     max_tokens_value = new_limit
                                     def _docs_for_rag_chain_lambda_retoken(query_text: str) -> List[Document]:
                                         docs, _ = get_final_documents_for_turn(query_text, vector_db, reranker_active_for_session and reranker_model_instance is not None, param_k_initial_retrieval, reranker_model_instance, param_base_similarity_threshold_session_default, param_reranker_score_threshold_session_default)
                                         return docs
                                     rag_chain = ( RunnablePassthrough.assign(retrieved_docs=itemgetter("question") | RunnableLambda(_docs_for_rag_chain_lambda_retoken)) | RunnablePassthrough.assign( context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs", []))), history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question", "")}).get("history",""))) | prompt | llm | StrOutputParser())
                                     follow_up_chain = ( RunnablePassthrough.assign(history=RunnableLambda(lambda x: memory.load_memory_variables({"question":x.get("question","")}).get("history",""))) | follow_up_prompt | llm | StrOutputParser())
                                     print(f"Max tokens set to {max_tokens_value}. RAG chain updated.")
                                 else: print("Failed to update token limit (LLM re-initialization error).")
                             else: print("Value must be > 50.")
                         except (TypeError, ValueError) as e: print(f"Invalid number. Usage: /set_tokens <number>. Error: {e}")
                else: print(f"Unknown command: {cmd}")
                turn_count -=1; continue

            current_turn_final_docs: List[Document] = []
            log_status = "N/A"; display_info = ""

            if not is_follow_up:
                print("Retrieving context...")
                try:
                    # Pass the SESSION DEFAULT thresholds to the function
                    # The function will use them as defaults if interactive prompt is used
                    current_turn_final_docs, display_info = get_final_documents_for_turn( 
                        user_input, vector_db,
                        reranker_active_for_session and reranker_model_instance is not None,
                        param_k_initial_retrieval,
                        reranker_model_instance,
                        param_base_similarity_threshold_session_default, 
                        param_reranker_score_threshold_session_default 
                    )
                    last_final_docs_for_follow_up = current_turn_final_docs 
                    log_status = f"Success ({len(current_turn_final_docs)} final docs)"
                except Exception as e:
                    print(f"\n[Error] Retrieval failed: {e}"); traceback.print_exc()
                    current_turn_final_docs = []; log_status = f"Error: {type(e).__name__}"
            else: 
                current_turn_final_docs = last_final_docs_for_follow_up if last_final_docs_for_follow_up else []
                if current_turn_final_docs: 
                    doc_count = len(current_turn_final_docs)
                    # display_info for follow-up might need to reflect the threshold used for those docs
                    # For now, keeping it simple
                    if reranker_active_for_session and reranker_model_instance:
                         display_info = f"Using previous context ({doc_count} docs)"
                    else:
                         display_info = f"Using previous context ({doc_count} docs)"
                else:
                    display_info = "Using previous context (none available)"
                log_status = "Success (Used Previous Context)" if current_turn_final_docs else "Failed (No Previous Context)"


            if show_chunks_in_terminal:
                display_retrieved_docs(user_input, current_turn_final_docs, display_info)
            else:
                print(f"[*] Retrieved {len(current_turn_final_docs)} final docs ({display_info}, Display disabled).")

            context_details_log = []
            if current_turn_final_docs:
                for i, doc in enumerate(current_turn_final_docs):
                    s_meta = {} 
                    for k_meta, v_meta in doc.metadata.items():
                        if isinstance(v_meta, (dict,list,set)): 
                            try: s_meta[k_meta] = json.dumps(v_meta) 
                            except TypeError: s_meta[k_meta] = str(v_meta) 
                        else: s_meta[k_meta] = v_meta
                    score = doc.metadata.get('final_score_used_for_filtering') 
                    if score is not None: s_meta['logged_score'] = float(score) if isinstance(score, (float, int, np.floating, np.integer)) else str(score)
                    context_details_log.append({'idx':i+1, 'content':doc.page_content, 'metadata':s_meta})

            ai_response = "N/A (Retrieval Only or LLM Error)"
            if session_mode == "rag":
                if not llm: ai_response = "Error: LLM not initialized."
                elif not current_turn_final_docs and not is_follow_up : ai_response = "Error: No context after filtering for LLM." 
                elif not current_turn_final_docs and is_follow_up and not last_final_docs_for_follow_up: ai_response = "Error: No previous context for follow-up."
                else:
                    print(f"\nAI ({selected_model_name}): ", end="")
                    full_resp = ""
                    try:
                        chain_input = {"question": user_input}
                        active_chain = rag_chain
                        if is_follow_up:
                             if last_final_docs_for_follow_up: 
                                chain_input["context"] = format_docs_for_llm(last_final_docs_for_follow_up) 
                                active_chain = follow_up_chain 
                             else: 
                                active_chain = None 
                                full_resp = "(Cannot follow-up, no previous context available)"
                                print(full_resp)

                        if active_chain: 
                            for chunk_resp in active_chain.stream(chain_input):
                                sys.stdout.write(chunk_resp); sys.stdout.flush(); full_resp += chunk_resp
                    except Exception as e: print(f"\n[Error in RAG chain]: {e}"); traceback.print_exc(); full_resp = "(LLM Error)"
                    print(); ai_response = full_resp if full_resp.strip() else "(Empty LLM response)"
                    if llm and (not is_follow_up or (is_follow_up and last_final_docs_for_follow_up)): 
                         memory.save_context({"question": user_input}, {"ai_response": ai_response})

            conversation_log.append({
                "turn": turn_count, "mode": session_mode, "is_follow_up": is_follow_up,
                "input": user_input, "retrieval_status": log_status, "retrieval_info_str": display_info,
                "docs_count": len(context_details_log), "docs_details": context_details_log,
                "ai_response": ai_response
            })
    except (EOFError, KeyboardInterrupt): print("\n\nUser interrupted.")
    except Exception as e: print(f"\n--- Error in main loop: {e} ---"); traceback.print_exc()
    finally: 
        if conversation_log:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}_{ts}.json")
                summary = {
                    "start_time": ts, "mode": session_mode, "llm_used":selected_model_name,
                    "reranker_active_and_available": reranker_active_for_session and reranker_model_instance is not None,
                    "k_initial_retrieval_for_session": param_k_initial_retrieval, 
                    "reranker_score_threshold_session_default_if_active": param_reranker_score_threshold_session_default if reranker_active_for_session and reranker_model_instance else "N/A",
                    "base_similarity_threshold_session_default_if_direct": param_base_similarity_threshold_session_default if not (reranker_active_for_session and reranker_model_instance) else "N/A",
                    "db_path": os.path.abspath(DB_DIR), "total_turns": len(conversation_log)
                }
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump({"summary":summary, "turns":conversation_log}, f, indent=4, default=str)
                print(f"\n[*] Log saved to {log_file}")
            except Exception as e: print(f"\n[Error saving log]: {e}")
        else: print("\n[*] No turns to log.")
        print("\nChat session finished.")