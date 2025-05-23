# D:\YouTubeTranscriptScraper\scripts\query_rag_database.py
# This will be based on the "original that was working" file you provided.
import os
import sys
import json
import torch
import traceback
import argparse
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Optional
import requests # For check_ollama_connection

# Langchain Imports
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.retrievers import ContextualCompressionRetriever # Standard import path
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
    from langchain_core.documents import Document
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    print(f"[Error] Failed to import core LangChain components: {e}")
    print("Please ensure 'langchain-chroma', 'langchain-huggingface', 'langchain', 'langchain-core' are installed.")
    sys.exit(1)

# Re-ranking Imports (Optional)
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    RERANKING_AVAILABLE = True
except ImportError:
    print("[Warning] Re-ranking components not found. Re-ranking will be disabled.")
    print("         Attempted to import CrossEncoderReranker from 'langchain.retrievers.document_compressors'")
    print("         and HuggingFaceCrossEncoder from 'langchain_community.cross_encoders'.")
    print("         If re-ranking is desired, please check your LangChain installation and import paths.")
    CrossEncoderReranker = None
    HuggingFaceCrossEncoder = None
    RERANKING_AVAILABLE = False

# LLM Import (Try multiple paths)
try:
    from langchain_community.llms import Ollama as OllamaLLM
except ImportError:
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
         print("[Error] Failed to import Ollama LLM. Install 'langchain-community' or 'langchain-ollama'.")
         sys.exit(1)


# --- Determine Script and Project Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
PROJECT_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"[*] Query Script directory: {SCRIPT_DIR}")
print(f"[*] Query Determined Project root: {PROJECT_ROOT_DIR}")

# --- Configuration (Using New Structure & Defaults) ---
# DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT_DIR, "database", "chroma_db_multi_source") # Original from your script
DEFAULT_DB_DIR = "C:\\YouTubeTranscriptScraper\\database\\chroma_db_multi_source" # Your confirmed working SSD path
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "output") # For chat logs
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # Must match ingest

DEFAULT_RERANKER_ENABLED = RERANKING_AVAILABLE # Default to available if import worked
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES_TO_RETRIEVE = 10 # k for initial retrieval for reranker, or k for direct retrieval
DEFAULT_FINAL_DOCUMENTS_TO_LLM = 5  # n for final documents after rerank

# AVAILABLE_MODELS from your script (ensure this matches what you want)
AVAILABLE_MODELS = [
    "granite3.3:2b", "phi4-mini-reasoning", "dolphin3:8b",
    "llama3.1:8b", "qwen3:4b", "gemma3:4b"
]
# Deduplicate and sort for cleaner display if you add more common ones
AVAILABLE_MODELS = sorted(list(set(AVAILABLE_MODELS + [
    "llama3:latest", "mistral:latest", "codellama:latest", "phi3:latest", 
    "qwen:latest", "gemma:latest"
])))


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
COMMAND_KEYWORDS = {"/set_tokens", "/help", "/mode", "/showchunks"}
DEFAULT_SESSION_MODE = "rag"
DEFAULT_SHOW_CHUNKS = True

def get_device():
    if torch.cuda.is_available():
        try:
             _ = torch.tensor([1.0]).to('cuda')
             return "cuda"
        except Exception as e: print(f"[Warning] CUDA found but failed to initialize: {e}. Falling back to CPU.")
    return "cpu"

def initialize_embeddings(device_to_use):
    print(f"[*] Loading embedding model '{EMBEDDING_MODEL_NAME}' onto device '{device_to_use}'...")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device_to_use},
            encode_kwargs={'normalize_embeddings': True},
            show_progress=False
        )
        print("[*] Performing test embedding...")
        _ = embedding_function.embed_query("test")
        print("[*] Embedding model loaded and tested successfully.")
        return embedding_function
    except Exception as e:
        print(f"[ERROR] initializing embedding model on device '{device_to_use}': {e}"); traceback.print_exc(); sys.exit(1)

def initialize_db(db_dir_to_use, embedding_func_to_use):
    print(f"[*] Initializing/Loading ChromaDB from: '{db_dir_to_use}'")
    if not os.path.isdir(db_dir_to_use): # Check if path exists
        print(f"[Error] ChromaDB directory not found: {db_dir_to_use}. Please ensure it exists or run ingestion first.")
        sys.exit(1)
    try:
        vector_db = Chroma(persist_directory=db_dir_to_use, embedding_function=embedding_func_to_use)
        count = vector_db._collection.count()
        print(f"[*] Connected to DB. Collection contains {count} documents.")
        return vector_db
    except Exception as e:
        print(f"[ERROR] initializing Chroma DB from '{db_dir_to_use}': {e}"); traceback.print_exc(); sys.exit(1)

def format_docs_for_llm(docs: List[Document]) -> str:
    if not isinstance(docs, list) or not docs: return "No relevant documents found in the database for this query."
    formatted = []
    for i, doc in enumerate(docs):
        if isinstance(doc, Document) and hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            content = doc.page_content; metadata = doc.metadata
            source_type = metadata.get('source_type', 'Unknown')
            title = metadata.get('title') or metadata.get('vulnerabilityName') or metadata.get('pdf_title') or metadata.get('filename', 'N/A')
            chunk_num = metadata.get('chunk_number', 'N/A')
            score = metadata.get('relevance_score'); score_info = f" (Relevance: {score:.4f})" if score is not None else ""
            identifier = f" (CVE: {metadata.get('cveID')})" if metadata.get('cveID') else f" (Video: {metadata.get('video_id')})" if metadata.get('video_id') else ""
            formatted.append(f"--- Context Chunk {i+1} (Source: {source_type}{identifier}, Title: '{title}', Chunk: {chunk_num}){score_info} ---\n{content}")
        else: print(f"[Warning] format_docs: unexpected item type: {type(doc)}"); formatted.append(f"--- Context Chunk {i+1} ---\n{str(doc)}")
    return "\n\n".join(formatted) if formatted else "No valid documents retrieved or formatted."

def check_ollama_connection(model_name_to_check=""):
    try:
        try: import ollama
        except ImportError: raise ImportError("Python 'ollama' library not found. Run 'pip install ollama'.")
        list_response = ollama.list()
        if not isinstance(list_response, dict) or 'models' not in list_response: print("[Warning] Unexpected response format from Ollama list."); return True
        if model_name_to_check:
            models_list = list_response.get('models', [])
            if not isinstance(models_list, list): print("[Warning] Could not parse Ollama model list."); return True
            model_found = any(isinstance(m, dict) and isinstance(m.get('name'), str) and (m['name'] == model_name_to_check or m['name'].startswith(model_name_to_check + ':')) for m in models_list)
            if model_found: return True
            else:
                 base_name_exists = False
                 if ':' not in model_name_to_check: base_name_exists = any(isinstance(m, dict) and isinstance(m.get('name'), str) and m['name'].startswith(model_name_to_check + ':') for m in models_list)
                 if base_name_exists: print(f"[Info] Exact tag '{model_name_to_check}' not found, but related tags exist. Assuming usable."); return True
                 else: print(f"[Error] Model '{model_name_to_check}' not found. Pull with: 'ollama pull {model_name_to_check}'"); return False
        return True
    except ImportError as e: print(f"[Error] {e}"); return False
    except requests.exceptions.ConnectionError: print("\n[Error] Connection to Ollama server failed. Is Ollama running?"); return False
    except Exception as e: print(f"\n[Error] checking Ollama: {type(e).__name__}: {e}"); return False

def display_retrieved_docs(query: str, docs: List[Document], k_retrieved_info: str): # Modified to accept k_retrieved_info
    print(f"\n--- Top {len(docs)} Retrieved Docs for Query: '{query}' ({k_retrieved_info}) ---")
    if not docs: print("--- No documents found. ---"); return
    for i, doc in enumerate(docs):
        if isinstance(doc, Document):
            metadata = doc.metadata; content = doc.page_content; score = metadata.get('relevance_score')
            content_snippet = content.replace('\n', ' ').strip()[:250] + ('...' if len(content) > 250 else '')
            score_info = f"Relevance: {score:.4f} | " if score is not None else ""
            source_type = metadata.get('source_type', 'Unknown')
            title = metadata.get('title') or metadata.get('vulnerabilityName') or metadata.get('pdf_title') or metadata.get('filename', 'N/A')
            chunk_num = metadata.get('chunk_number', 'N/A')
            identifier = f" (CVE: {metadata.get('cveID')})" if metadata.get('cveID') else f" (Video: {metadata.get('video_id')})" if metadata.get('video_id') else ""
            print(f"[{i+1}] {score_info}Source: {source_type}{identifier} | Title: {title} | Chunk: {chunk_num}")
            print(f"    Snippet: {content_snippet}")
        else: print(f"[{i+1}] Unexpected item type: {type(doc)}"); print(f"    Content: {str(doc)}")
    print("-" * 80)

def initialize_llm(model_name_to_init: str, num_tokens_to_set: int) -> Optional[OllamaLLM]:
    print(f"\n[*] Initializing Ollama LLM: '{model_name_to_init}' with max_tokens={num_tokens_to_set}...")
    try:
        llm_instance = OllamaLLM(model=model_name_to_init, num_predict=num_tokens_to_set)
        print("[*] Testing LLM response...")
        # A more robust test, expecting a simple string output
        test_output = llm_instance.invoke("Briefly, what is 1+1? Respond with only the answer.")
        if not isinstance(test_output, str) or not test_output.strip():
             print(f"[Warning] LLM test response was empty or not a string. Output: {test_output}")
        print("[*] LLM initialized successfully.")
        return llm_instance
    except Exception as e: print(f"[ERROR] initializing Ollama LLM '{model_name_to_init}': {e}"); traceback.print_exc(); return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chat Application using ChromaDB and Ollama.")
    parser.add_argument("--db_dir", default=None, help=f"Path to the ChromaDB directory (Optional, overrides default from script: {DEFAULT_DB_DIR}).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save chat logs (default: {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args()

    DB_DIR = args.db_dir if args.db_dir else DEFAULT_DB_DIR # Use CLI arg if provided, else script default
    OUTPUT_DIR = args.output_dir

    print("Initializing Chat Session...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[*] Checking Ollama connection...")
    if not check_ollama_connection(): sys.exit(1)
    print("[*] Ollama connection OK.")

    embed_device_name = get_device()
    print(f"[*] Using device for embeddings: {embed_device_name}")
    embedding_function = initialize_embeddings(embed_device_name)
    if embedding_function is None: sys.exit(1)
    vector_db = initialize_db(DB_DIR, embedding_function) # Use effective DB_DIR
    if vector_db is None: sys.exit(1)

    session_mode = DEFAULT_SESSION_MODE
    show_chunks_in_terminal = DEFAULT_SHOW_CHUNKS
    selected_model_name = None; llm = None; max_tokens_value = DEFAULT_MAX_TOKENS
    reranker_enabled_session_flag = DEFAULT_RERANKER_ENABLED
    candidates_k_value = DEFAULT_CANDIDATES_TO_RETRIEVE
    final_top_n_value = DEFAULT_FINAL_DOCUMENTS_TO_LLM

    print("\n--- Select Session Mode ---")
    print("1. RAG Mode (Retrieve + AI Response)")
    print("2. Retrieval Only Mode (Show Retrieved Chunks, No AI)")
    while True:
        mode_choice = input(f"Choose mode (1 or 2, Enter for default='{1 if DEFAULT_SESSION_MODE == 'rag' else 2}'): ").strip()
        if not mode_choice: session_mode = DEFAULT_SESSION_MODE; break
        elif mode_choice == '1': session_mode = "rag"; break
        elif mode_choice == '2': session_mode = "retrieval_only"; break
        else: print("Invalid choice.")
    print(f"[*] Running in {session_mode.replace('_', ' ').title()} mode.")

    print("\n--- Configure Display Options ---")
    while True:
        show_chunks_choice = input(f"Show retrieved chunk details in terminal? (Y/n, Enter for default={'Y' if DEFAULT_SHOW_CHUNKS else 'N'}): ").strip().lower()
        if not show_chunks_choice: show_chunks_in_terminal = DEFAULT_SHOW_CHUNKS; break
        elif show_chunks_choice == 'y': show_chunks_in_terminal = True; break
        elif show_chunks_choice == 'n': show_chunks_in_terminal = False; break
        else: print("Invalid input.")
    print(f"[*] Chunk display in terminal: {'ENABLED' if show_chunks_in_terminal else 'DISABLED'}")

    if session_mode == "rag":
        print("\n--- Select LLM Model ---")
        print("Available models (some common, some from your list):")
        for i, model_name_option in enumerate(AVAILABLE_MODELS): print(f"  {i+1}. {model_name_option}")
        while True:
            try:
                choice = input(f"Choose model number (1-{len(AVAILABLE_MODELS)}), or type name: ").strip()
                if not choice and AVAILABLE_MODELS: temp_model_name = AVAILABLE_MODELS[0]; print(f"Defaulting to {temp_model_name}")
                elif choice.isdigit():
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(AVAILABLE_MODELS): temp_model_name = AVAILABLE_MODELS[model_index]
                    else: print("Invalid choice number."); continue
                elif choice: temp_model_name = choice # Allow typing model name
                else: print("No models available or invalid input."); continue

                if check_ollama_connection(model_name_to_check=temp_model_name): selected_model_name = temp_model_name; break
                else: print(f"Model '{temp_model_name}' not available locally or connection issue. Please ensure it's pulled or choose another.")
            except ValueError: print("Invalid input for model number.")
            except IndexError: print("No models in list to default to.")

        print(f"[*] Using LLM: {selected_model_name}")
        while True:
            max_tokens_input = input(f"[*] Initial max AI response tokens? (Enter={DEFAULT_MAX_TOKENS}): ").strip()
            if not max_tokens_input: max_tokens_value = DEFAULT_MAX_TOKENS; break 
            try:
                max_tokens_value = int(max_tokens_input);
                if max_tokens_value > 50: break 
                else: print("Please enter a value greater than 50.")
            except ValueError: print("Invalid input. Please enter a number.")
        llm = initialize_llm(selected_model_name, max_tokens_value)
        if llm is None: print("[Error] Failed to initialize LLM. RAG mode will be impaired.")
    else:
        selected_model_name = "N/A (Retrieval Only)"; max_tokens_value = 0

    print("\n--- Configure Retrieval/Re-ranking ---")
    if RERANKING_AVAILABLE:
        while True:
            enable_choice = input(f"Enable Re-ranking? (Y/n, Enter={'Y' if DEFAULT_RERANKER_ENABLED else 'N'}): ").strip().lower()
            if not enable_choice: reranker_enabled_session_flag = DEFAULT_RERANKER_ENABLED; break
            elif enable_choice == 'y': reranker_enabled_session_flag = True; break
            elif enable_choice == 'n': reranker_enabled_session_flag = False; break
            else: print("Invalid input.")
    else:
        print("[Info] Re-ranking components not found or import failed. Re-ranking is DISABLED.")
        reranker_enabled_session_flag = False

    if reranker_enabled_session_flag: # Only ask these if reranking is conceptually enabled AND available
        print("Re-ranking is ON. Configure initial candidates and final documents for re-ranker.")
        while True:
            cand_k_input = input(f"Initial candidates to fetch for re-ranker (k)? (Default={DEFAULT_CANDIDATES_TO_RETRIEVE}): ").strip()
            if not cand_k_input: candidates_k_value = DEFAULT_CANDIDATES_TO_RETRIEVE; break
            try: 
                candidates_k_value = int(cand_k_input)
                if candidates_k_value > 0: break
                else: print("Enter > 0.")
            except ValueError: print("Invalid input.") # SyntaxError was here
        
        while True:
            final_n_input = input(f"Final documents re-ranker should output (top_n)? (Default={DEFAULT_FINAL_DOCUMENTS_TO_LLM}): ").strip()
            if not final_n_input: final_top_n_value = DEFAULT_FINAL_DOCUMENTS_TO_LLM; break
            try: 
                final_top_n_value = int(final_n_input)
                if final_top_n_value > 0: break
                else: print("Enter > 0.")
            except ValueError: print("Invalid input.")
        if final_top_n_value > candidates_k_value: 
            print(f"Warning: Final docs ({final_top_n_value}) > Initial candidates ({candidates_k_value}). Setting Final=Initial.")
            final_top_n_value = candidates_k_value
    else: 
        print("Re-ranking is OFF. Configure number of documents for direct retrieval.")
        while True:
            final_n_input = input(f"Documents to retrieve directly (k)? (Default={DEFAULT_FINAL_DOCUMENTS_TO_LLM}): ").strip()
            if not final_n_input: final_top_n_value = DEFAULT_FINAL_DOCUMENTS_TO_LLM; break
            try: 
                final_top_n_value = int(final_n_input)
                if final_top_n_value > 0: break
                else: print("Enter > 0.")
            except ValueError: print("Invalid input.")
        candidates_k_value = final_top_n_value # If no reranking, initial_k for base retriever effectively becomes this.

    reranker_compressor = None
    if reranker_enabled_session_flag and RERANKING_AVAILABLE and CrossEncoderReranker and HuggingFaceCrossEncoder:
        print(f"[*] Initializing Re-ranker model '{DEFAULT_RERANKER_MODEL_NAME}' on device '{embed_device_name}'...")
        try:
            encoder_wrapper = HuggingFaceCrossEncoder(model_name=DEFAULT_RERANKER_MODEL_NAME, model_kwargs={'device': embed_device_name})
            reranker_compressor = CrossEncoderReranker(model=encoder_wrapper, top_n=final_top_n_value) # top_n is how many docs compressor outputs
            print(f"[*] Re-ranker initialized (Will keep Top {final_top_n_value} from initial candidates).")
        except Exception as e: 
            print(f"[Error] initializing Re-ranker: {e}"); traceback.print_exc()
            print("[Warning] Proceeding without re-ranking due to initialization error."); reranker_enabled_session_flag = False 
    
    # --- MODIFIED RETRIEVER SETUP ---
    if reranker_enabled_session_flag and reranker_compressor:
        # Base retriever for compression fetches 'candidates_k_value'
        base_retriever_for_compression = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": candidates_k_value})
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor, 
            base_retriever=base_retriever_for_compression
        )
        print(f"[*] Using Re-ranking Retriever (Initial fetch k={candidates_k_value}, re-ranker output top_n={final_top_n_value}).")
        retriever_info_for_display_prefix = f"Fetched initial k={candidates_k_value}, Re-ranked to top "
    else:
        # No re-ranking (either disabled, unavailable, or failed to init)
        # The retriever should fetch 'final_top_n_value' documents directly.
        final_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": final_top_n_value})
        print(f"[*] Using Base Similarity Retriever (k={final_top_n_value}).")
        retriever_info_for_display_prefix = f"Retrieved k="
        if reranker_enabled_session_flag and not reranker_compressor: 
            print("[Warning] Re-ranking was selected but compressor is not active. Fell back to Base Similarity Retriever.")
    # --- END MODIFIED RETRIEVER SETUP ---


    memory = ConversationBufferMemory(memory_key="history", input_key="question", output_key="ai_response", return_messages=False)
    rag_chain, follow_up_chain = None, None
    prompt = PromptTemplate(template=MEMORY_RAG_PROMPT_TEMPLATE, input_variables=["history", "context", "question"])
    follow_up_prompt = PromptTemplate(template=MEMORY_RAG_PROMPT_TEMPLATE, input_variables=["history", "context", "question"]) 

    if session_mode == "rag" and llm:
        def get_docs_for_rag_chain(query: str) -> List[Document]:
            try: 
                retrieved = final_retriever.invoke(query)
                return retrieved if isinstance(retrieved, list) else []
            except Exception as retrieve_err: 
                print(f"\n[Error during retrieval in get_docs_for_rag_chain]: {retrieve_err}")
                return []

        rag_chain = (
            RunnablePassthrough.assign(
                retrieved_docs=itemgetter("question") | RunnableLambda(get_docs_for_rag_chain)
            )
            | RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs", []))),
                history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question", "")})["history"])
            )
            | prompt | llm | StrOutputParser()
        )
        print("[*] Main RAG chain created.")
        
        follow_up_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question", "")})["history"])
            )
            | follow_up_prompt | llm | StrOutputParser()
        )
        print("[*] Follow-up chain pattern created.")

    print("\n" + "="*30 + " Chat Session Started " + "="*30)
    print(f"Session Mode: {session_mode.replace('_', ' ').title()}")
    if session_mode == "rag": print(f"LLM Model: {selected_model_name} | Max Tokens: {max_tokens_value}")
    
    if reranker_enabled_session_flag and reranker_compressor:
        print(f"Re-ranking: ENABLED (Fetch initial k={candidates_k_value}, Re-ranker output top_n={final_top_n_value})")
    else:
        print(f"Re-ranking: DISABLED (Directly retrieving top k={final_top_n_value})")
    
    print(f"Show Chunks in Terminal: {'ENABLED' if show_chunks_in_terminal else 'DISABLED'}")
    print(f"Type '/help' for commands, '{', '.join(EXIT_KEYWORDS)}' to end.")
    print("="* (62 + len(" Chat Session Started ") - 2)) # Adjusted length

    conversation_log = []
    turn_count = 0
    last_retrieved_docs: Optional[List[Document]] = None

    try:
        while True:
            turn_count += 1; is_follow_up_on_last_context = False
            
            if turn_count > 1 and session_mode == "rag" and last_retrieved_docs and llm:
                while True:
                    follow_up_choice = input("Follow-up on last retrieved context? (y/n, Enter=n): ").strip().lower()
                    if not follow_up_choice or follow_up_choice == 'n': is_follow_up_on_last_context = False; break
                    elif follow_up_choice == 'y': is_follow_up_on_last_context = True; print("[Using previous context for this follow-up question]"); break
                    else: print("Invalid input.")

            user_input_query = input(f"\n[{turn_count}] You: ").strip()
            if not user_input_query: turn_count -= 1; continue

            if user_input_query.lower() in EXIT_KEYWORDS: print("\nExiting..."); break
            if user_input_query.startswith('/'):
                 command_parts = user_input_query.lower().split()
                 command = command_parts[0]
                 # --- COMMAND HANDLING (copied from your provided original script) ---
                 if command == '/help':
                     print("\nAvailable commands:")
                     print("  /mode         : Toggle RAG/Retrieval Only (prompts LLM re-selection if switching to RAG).")
                     print("  /showchunks   : Toggle display of retrieved chunk details.")
                     print("  /set_tokens   : Change AI response token limit (RAG mode only, re-initializes LLM).")
                     print(f"  {', '.join(EXIT_KEYWORDS)} : Exit.")
                     print("  /help         : Show this help message.")
                 elif command == '/mode':
                     current_session_mode_before_switch = session_mode
                     session_mode = "retrieval_only" if session_mode == "rag" else "rag"
                     print(f"Switched to {session_mode.replace('_', ' ').title()} mode.")
                     if session_mode == "rag" and (not llm or current_session_mode_before_switch == "retrieval_only"): 
                          print("LLM needs to be selected/re-initialized for RAG mode.")
                          print("\n--- Select LLM Model ---"); print("Available models:")
                          for i, model_name_option in enumerate(AVAILABLE_MODELS): print(f"  {i+1}. {model_name_option}")
                          while True:
                              try:
                                  choice = input(f"Choose model number (1-{len(AVAILABLE_MODELS)}), or type name: ").strip()
                                  if not choice and AVAILABLE_MODELS: temp_model_name = AVAILABLE_MODELS[0]; print(f"Defaulting to {temp_model_name}")
                                  elif choice.isdigit():
                                      model_index = int(choice) - 1
                                      if 0 <= model_index < len(AVAILABLE_MODELS): temp_model_name = AVAILABLE_MODELS[model_index]
                                      else: print("Invalid choice number."); continue
                                  elif choice: temp_model_name = choice
                                  else: print("No models available or invalid input."); continue
                                  if check_ollama_connection(model_name_to_check=temp_model_name): selected_model_name = temp_model_name; break
                                  else: print(f"Model '{temp_model_name}' not available.")
                              except ValueError: print("Invalid input.")
                          print(f"[*] Using LLM: {selected_model_name}")
                          while True: 
                              max_tokens_input = input(f"[*] Max AI response tokens? (Enter={DEFAULT_MAX_TOKENS}): ").strip()
                              if not max_tokens_input: max_tokens_value = DEFAULT_MAX_TOKENS; break
                              try:
                                  max_tokens_value = int(max_tokens_input)
                                  if max_tokens_value > 50: break
                                  else: print("Please enter > 50.")
                              except ValueError: print("Invalid input.")
                          llm = initialize_llm(selected_model_name, max_tokens_value)
                          if llm is None: print("[Warning] LLM init failed. RAG mode may not function fully.")
                          else: 
                            rag_chain = ( RunnablePassthrough.assign(retrieved_docs=itemgetter("question") | RunnableLambda(get_docs_for_rag_chain)) | RunnablePassthrough.assign( context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs",[]))), history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question","")})["history"]) ) | prompt | llm | StrOutputParser() )
                            follow_up_chain = ( RunnablePassthrough.assign( history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question","")})["history"]) ) | follow_up_prompt | llm | StrOutputParser() )
                            print("[*] RAG chains re-created with new LLM settings.")
                 elif command == '/showchunks':
                     show_chunks_in_terminal = not show_chunks_in_terminal
                     print(f"Chunk display in terminal: {'ENABLED' if show_chunks_in_terminal else 'DISABLED'}")
                 elif command == '/set_tokens':
                     if session_mode != "rag": print("Token limit only applicable in RAG mode.")
                     elif not llm or not selected_model_name : print("LLM not initialized. Select RAG mode and an LLM first.")
                     else:
                         print(f"Current max tokens: {max_tokens_value}")
                         while True:
                             new_limit_str = input("Enter new max token limit (> 50): ").strip()
                             try:
                                 new_limit = int(new_limit_str)
                                 if new_limit > 50:
                                     new_llm_instance = initialize_llm(selected_model_name, new_limit)
                                     if new_llm_instance:
                                         llm = new_llm_instance; max_tokens_value = new_limit
                                         rag_chain = ( RunnablePassthrough.assign(retrieved_docs=itemgetter("question") | RunnableLambda(get_docs_for_rag_chain)) | RunnablePassthrough.assign( context=RunnableLambda(lambda x: format_docs_for_llm(x.get("retrieved_docs",[]))), history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question","")})["history"]) ) | prompt | llm | StrOutputParser() )
                                         follow_up_chain = ( RunnablePassthrough.assign(history=RunnableLambda(lambda x: memory.load_memory_variables({"question": x.get("question","")})["history"]))| follow_up_prompt | llm | StrOutputParser())
                                         print(f"Max tokens set to {max_tokens_value}. RAG chains updated.")
                                     else: print("Failed to update token limit (LLM re-initialization error).")
                                     break
                                 else: print("Value must be > 50.")
                             except ValueError: print("Invalid number.")
                 else: print(f"Unknown command: {command}. Type /help for options.")
                 turn_count -= 1; continue 


            current_docs_for_this_specific_turn = None
            log_status_for_turn = "N/A"
            display_k_info_for_log_this_turn = "" 

            if not is_follow_up_on_last_context: 
                print("Retrieving context for new query...")
                try:
                    current_docs_for_this_specific_turn = final_retriever.invoke(user_input_query)
                    last_retrieved_docs = current_docs_for_this_specific_turn 
                    log_status_for_turn = f"Success ({len(current_docs_for_this_specific_turn or [])} docs)"
                    # Set info for display_retrieved_docs
                    if reranker_enabled_session_flag and reranker_compressor:
                        display_k_info_for_log_this_turn = f"Fetched initial k={candidates_k_value}, Re-ranked to top {len(current_docs_for_this_specific_turn or [])}"
                    else:
                        display_k_info_for_log_this_turn = f"Retrieved k={len(current_docs_for_this_specific_turn or [])}"
                except Exception as retrieve_err:
                    print(f"\n[Error] Retrieval failed for new query: {retrieve_err}"); traceback.print_exc()
                    current_docs_for_this_specific_turn = [] 
                    log_status_for_turn = f"Error: {type(retrieve_err).__name__}"
                
                if show_chunks_in_terminal: 
                    display_retrieved_docs(user_input_query, current_docs_for_this_specific_turn or [], display_k_info_for_log_this_turn)
                else: 
                    print(f"[*] Retrieved {len(current_docs_for_this_specific_turn or [])} chunks ({display_k_info_for_log_this_turn}, Display disabled).")
            
            else: 
                current_docs_for_this_specific_turn = last_retrieved_docs 
                if reranker_enabled_session_flag and reranker_compressor:
                    display_k_info_for_log_this_turn = f"Using previous context (originally k={candidates_k_value}, re-ranked to {len(current_docs_for_this_specific_turn or [])})"
                else:
                    display_k_info_for_log_this_turn = f"Using previous context (originally retrieved k={len(current_docs_for_this_specific_turn or [])})"
                if show_chunks_in_terminal: 
                    print("[Displaying previous context for follow-up query]")
                    display_retrieved_docs(user_input_query + " (follow-up on previous context)", current_docs_for_this_specific_turn or [], display_k_info_for_log_this_turn)
                else: 
                    print(f"[*] Using previous context ({len(current_docs_for_this_specific_turn or [])} chunks, display disabled for follow-up).")
                log_status_for_turn = "Success (Used Previous Context)" if current_docs_for_this_specific_turn else "Failed (No Previous Context available for Follow-up)"
            
            context_details_for_log_turn = []
            if current_docs_for_this_specific_turn:
                 for doc_idx, doc_item in enumerate(current_docs_for_this_specific_turn):
                      if isinstance(doc_item, Document):
                           serializable_metadata = {}
                           for k_meta, v_meta in doc_item.metadata.items():
                               if isinstance(v_meta, (dict, list, set)): 
                                   try: serializable_metadata[k_meta] = json.dumps(v_meta) 
                                   except TypeError: serializable_metadata[k_meta] = str(v_meta) 
                               elif isinstance(v_meta, (str, int, float, bool)) or v_meta is None: serializable_metadata[k_meta] = v_meta
                               else: serializable_metadata[k_meta] = str(v_meta)
                           score = serializable_metadata.get('relevance_score')
                           if score is not None:
                               try: serializable_metadata['relevance_score'] = float(score) 
                               except (ValueError, TypeError): serializable_metadata['relevance_score'] = str(score) 
                           
                           details = {
                               'retrieved_doc_index': doc_idx + 1, 
                               'content': doc_item.page_content, # Log full content
                               'metadata': serializable_metadata
                           }
                           context_details_for_log_turn.append(details)
            else: 
                context_details_for_log_turn = "Retrieval Failed/Empty or No Previous Context for this turn"

            ai_response_for_log_turn = "N/A (Retrieval Only Mode or LLM/Context Error)"
            if session_mode == "rag":
                if not llm: print("AI: Cannot respond, LLM not initialized."); ai_response_for_log_turn = "Error: LLM missing."
                elif not current_docs_for_this_specific_turn and not is_follow_up_on_last_context: print("AI: Cannot respond, no context retrieved for new query."); ai_response_for_log_turn = "Error: Context missing for new query."
                elif not current_docs_for_this_specific_turn and is_follow_up_on_last_context: print("AI: Cannot respond, no previous context for follow-up."); ai_response_for_log_turn = "Error: Context missing for follow-up."
                else:
                    print(f"\nAI ({selected_model_name} | max_tokens={max_tokens_value}): ", end="")
                    full_response_from_llm = ""
                    context_string_for_llm_turn = format_docs_for_llm(current_docs_for_this_specific_turn)
                    try:
                        input_payload_for_llm = {"question": user_input_query, "context": context_string_for_llm_turn}
                        chain_to_invoke_llm_with = rag_chain if not is_follow_up_on_last_context else follow_up_chain
                        if chain_to_invoke_llm_with:
                             for chunk_resp in chain_to_invoke_llm_with.stream(input_payload_for_llm, config=RunnableConfig(max_concurrency=1)):
                                 sys.stdout.write(chunk_resp); sys.stdout.flush(); full_response_from_llm += chunk_resp
                        else: full_response_from_llm = "(Error: Appropriate chain not available)"; print(f"\n[ERROR] Chain not available. Follow-up={is_follow_up_on_last_context}")
                    except Exception as chain_execution_err: print(f"\n[ERROR IN CHAIN EXECUTION]: {chain_execution_err}"); traceback.print_exc(); full_response_from_llm = "(Error during LLM generation)"
                    print(); 
                    if not full_response_from_llm.strip(): full_response_from_llm = "(No text generated or empty response)"
                    ai_response_for_log_turn = full_response_from_llm
                    try: memory.save_context({"question": user_input_query}, {"ai_response": full_response_from_llm})
                    except Exception as memory_save_err: print(f"\n[Error saving context to memory]: {memory_save_err}")
            
            # --- MODIFIED LOGGING KEY AND REMOVED SLICING ---
            conversation_log.append({
                "turn": turn_count, 
                "mode": session_mode, 
                "is_follow_up_on_previous_context": is_follow_up_on_last_context,
                "user_input": user_input_query, 
                "retrieval_status_for_turn": log_status_for_turn,
                "retrieved_documents_count_for_turn": len(context_details_for_log_turn) if isinstance(context_details_for_log_turn, list) else 0,
                "retrieved_documents_details_for_turn": context_details_for_log_turn, # Store ALL details
                "ai_max_tokens_setting": max_tokens_value if session_mode == "rag" else "N/A",
                "ai_response": ai_response_for_log_turn
            })

    except (EOFError, KeyboardInterrupt): print("\n\nUser interrupted. Exiting...")
    except Exception as e: print(f"\n--- Unexpected error in main loop: {e} ---"); traceback.print_exc(); print("Attempting to save log...")
    finally:
        if conversation_log:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}_{timestamp}.json")
                final_completed_turns = len(conversation_log)
                
                # Updated session summary keys for clarity
                session_summary_k_setting_key = "initial_candidates_k_setting_for_reranker" if (reranker_enabled_session_flag and reranker_compressor) else "documents_retrieved_k_setting"
                session_summary_top_n_key = "final_docs_top_n_from_reranker" if (reranker_enabled_session_flag and reranker_compressor) else "N/A (direct retrieval)"

                session_summary = {
                    "session_start_time": timestamp, "final_mode": session_mode,
                    "llm_model_used": selected_model_name, "final_max_tokens_setting": max_tokens_value,
                    "reranker_initially_available": RERANKING_AVAILABLE, 
                    "reranker_session_enabled_and_active": reranker_enabled_session_flag and reranker_compressor is not None, 
                    session_summary_k_setting_key: candidates_k_value, # k used for initial fetch or direct
                    session_summary_top_n_key: final_top_n_value if (reranker_enabled_session_flag and reranker_compressor) else final_top_n_value, # top_n from reranker or direct k
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "embedding_device": embed_device_name, "db_directory": os.path.abspath(DB_DIR),
                    "show_chunks_setting_in_terminal": show_chunks_in_terminal, 
                    "total_recorded_interaction_pairs": final_completed_turns
                }
                full_log_data = {"session_summary": session_summary, "conversation_turns": conversation_log}
                
                print(f"\n[*] Attempting to save conversation log to: {log_filename} (This might be large)...")
                with open(log_filename, 'w', encoding='utf-8') as f: 
                    json.dump(full_log_data, f, indent=4, ensure_ascii=False, default=str)
                print(f"[*] Conversation log saved successfully.")
            except Exception as log_e: print(f"\n[Error] saving log: {log_e}"); traceback.print_exc()
        else: print("\n[*] No conversation turns to log.")
        print("\nChat session finished.")