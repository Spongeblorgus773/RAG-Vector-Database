# D:\YouTubeTranscriptScraper\scripts\NEw_query_rag_database_refactored.py
import os
import sys
import json
import torch
import traceback
import argparse
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Optional, Tuple, Any
import requests
import numpy as np

# Langchain Imports
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig, Runnable
    from langchain_core.documents import Document
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    print(f"[Error] Failed to import core LangChain components: {e}")
    sys.exit(1)

# Re-ranking Imports
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    RERANKING_COMPONENT_AVAILABLE = True
except ImportError:
    print("[Warning] Re-ranking components not found. Re-ranking will be disabled.")
    CrossEncoderReranker = None
    HuggingFaceCrossEncoder = None
    RERANKING_COMPONENT_AVAILABLE = False

# LLM Import
try:
    from langchain_community.llms import Ollama as OllamaLLM 
except ImportError:
    try:
        from langchain_ollama import OllamaLLM 
    except ImportError:
         print("[Error] Failed to import Ollama LLM. Install 'langchain-community' or 'langchain-ollama'.")
         sys.exit(1)

# --- Configuration Class ---
class ScriptConfig:
    def __init__(self, db_dir_override: Optional[str] = None, output_dir_override: Optional[str] = None):
        try:
            self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.SCRIPT_DIR = os.getcwd()
        self.PROJECT_ROOT_DIR = os.path.dirname(self.SCRIPT_DIR)

        self.DEFAULT_DB_DIR = db_dir_override or "C:\\YouTubeTranscriptScraper\\database\\chroma_db_multi_source"
        self.DEFAULT_OUTPUT_DIR = output_dir_override or os.path.join(self.PROJECT_ROOT_DIR, "output")
        
        self.EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
        self.DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        self.DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER = 50
        self.DEFAULT_K_FOR_DIRECT_RETRIEVAL = 20
        self.DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER = 0.0
        self.DEFAULT_BASE_SIMILARITY_THRESHOLD = 0.75

        self.AVAILABLE_MODELS = sorted(list(set([
            "llama3:latest", "granite3.3:2b", "phi4-mini-reasoning:latest", 
            "gemma3:4b", "qwen3:4b", "dolphin3:8b", "llama3.1:8b",
            "codellama:latest", "mistral:latest", "phi3:latest", "gemma:latest", "qwen:latest"
        ])))
        self.MEMORY_RAG_PROMPT_TEMPLATE_STR = """
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
        self.OUTPUT_FILE_PREFIX = "rag_chat_session"
        self.DEFAULT_MAX_TOKENS = 1024
        self.EXIT_KEYWORDS = {"end", "stop", "quit", "bye", "exit"}
        self.DEFAULT_SESSION_MODE = "rag" 
        self.DEFAULT_SHOW_CHUNKS = True
        self.RERANKER_INITIALLY_ENABLED_BY_USER = RERANKING_COMPONENT_AVAILABLE

        print(f"[*] Config: Script directory: {self.SCRIPT_DIR}")
        print(f"[*] Config: Project root: {self.PROJECT_ROOT_DIR}")
        print(f"[*] Config: DB directory: {self.DEFAULT_DB_DIR}")
        print(f"[*] Config: Output directory: {self.DEFAULT_OUTPUT_DIR}")

# --- Session State Class ---
class SessionState:
    def __init__(self, config: ScriptConfig):
        self.config = config
        self.mode: str = config.DEFAULT_SESSION_MODE
        self.show_chunks_in_terminal: bool = config.DEFAULT_SHOW_CHUNKS
        self.selected_llm_name: Optional[str] = None
        self.llm_instance: Optional[OllamaLLM] = None
        self.max_tokens: int = config.DEFAULT_MAX_TOKENS
        
        self.reranker_active_for_session: bool = config.RERANKER_INITIALLY_ENABLED_BY_USER and RERANKING_COMPONENT_AVAILABLE
        self.k_initial_retrieval: int = config.DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER if self.reranker_active_for_session else config.DEFAULT_K_FOR_DIRECT_RETRIEVAL
        
        self.reranker_score_threshold_session_default: float = config.DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER
        self.base_similarity_threshold_session_default: float = config.DEFAULT_BASE_SIMILARITY_THRESHOLD

        self.memory = ConversationBufferMemory(memory_key="history", input_key="question", output_key="ai_response", return_messages=False)
        self.rag_chain_template: Optional[PromptTemplate] = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=config.MEMORY_RAG_PROMPT_TEMPLATE_STR
        )
        self.follow_up_chain_template: Optional[PromptTemplate] = self.rag_chain_template
        
        self.rag_runnable: Optional[Runnable] = None
        self.follow_up_runnable: Optional[Runnable] = None

        self.last_retrieved_docs_for_follow_up: Optional[List[Document]] = None
        self.conversation_log: List[Dict[str, Any]] = []
        self.turn_count: int = 0

# --- Helper Functions ---
def get_device_info() -> str:
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).to('cuda')
            print(f"[*] CUDA GPU found: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except Exception as e:
            print(f"[Warning] CUDA initialization failed: {e}. Falling back to CPU.")
            return "cpu"
    print("[*] No compatible GPU detected. Using CPU.")
    return "cpu"

def check_ollama_connection(model_to_verify: Optional[str] = None) -> bool:
    try:
        import ollama
        client = ollama.Client() # Defaults should be fine since OLLAMA_HOST is not set by user for script

        if model_to_verify:
            print(f"\n[Debug] Verifying specific model '{model_to_verify}' with Ollama server...")
            try:
                client.show(model_to_verify) 
                print(f"[*] Ollama server acknowledged model '{model_to_verify}'. Connection and model seem OK.")
                return True
            except ollama.ResponseError as e: 
                 if e.status_code == 404:
                    print(f"[Error] Ollama server reports model '{model_to_verify}' not found (404).")
                 else:
                    print(f"[Error] Ollama API error when checking model '{model_to_verify}': Status {e.status_code} - {e.error}")
                 return False
            except requests.exceptions.ConnectionError:
                print(f"[Error] Ollama server connection failed when verifying model '{model_to_verify}'.")
                return False
            except Exception as e_show:
                print(f"[Error] Unexpected issue verifying model '{model_to_verify}' with Ollama: {type(e_show).__name__} - {e_show}")
                return False
        else: # Basic connectivity check
            try:
                client.list() # Just try to list, don't need to parse here anymore
                print("[*] Ollama connection successful (basic list command acknowledged).")
                return True
            except requests.exceptions.ConnectionError:
                print(f"[Error] Ollama server connection failed (basic list command).")
                return False
            except Exception as e_list:
                print(f"[Error] Issue during basic Ollama list command for connectivity check: {type(e_list).__name__} - {e_list}")
                return False
    except ImportError:
        print("[Error] Python 'ollama' library not found. Install with 'pip install ollama'.")
        return False
    except Exception as e_conn_general: 
        print(f"[Error] Ollama connection check encountered an unexpected issue: {type(e_conn_general).__name__} - {e_conn_general}")
        traceback.print_exc()
        return False

def display_retrieved_docs(query: str, docs: List[Document], info_str: str, show_snippets: bool = True):
    print(f"\n--- {len(docs)} Retrieved Docs for Query: '{query}' ---")
    print(f"    Retrieval Info: {info_str}")
    if not docs: print("--- No documents found matching criteria. ---"); return
    for i, doc in enumerate(docs):
        meta = doc.metadata; content = doc.page_content
        score_info = f"Score: {meta.get('final_score_used_for_filtering'):.4f} | " if meta.get('final_score_used_for_filtering') is not None else ""
        title = meta.get('title', meta.get('vulnerabilityName', meta.get('pdf_title', meta.get('filename', 'N/A'))))
        src = meta.get('source_type', 'Unknown'); chunk_idx = meta.get('chunk_number', meta.get('chunk_index', '?'))
        doc_id = meta.get('id_for_log', meta.get('id', f'doc_idx_{i}')); id_info = f"(ID: {doc_id})"
        print(f"[{i+1}] {score_info}Source: {src}{id_info} | Title: '{title}' | Chunk: {chunk_idx}")
        if show_snippets: print(f"    Snippet: {content[:250].replace(chr(10), ' ').strip()}{'...' if len(content)>250 else ''}")
    print("-" * 80)

def save_chat_log(session_state: SessionState, rag_manager: 'RAGManager', config: ScriptConfig):
    if not session_state.conversation_log: print("\n[*] No turns to log."); return
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{config.OUTPUT_FILE_PREFIX}_{ts}.json")
        summary = {
            "timestamp": ts, "mode": session_state.mode,
            "llm": session_state.selected_llm_name if session_state.mode == "rag" else "N/A",
            "max_tokens": session_state.max_tokens if session_state.mode == "rag" else "N/A",
            "reranker_avail": RERANKING_COMPONENT_AVAILABLE, "reranker_active": session_state.reranker_active_for_session,
            "reranker_model": config.DEFAULT_RERANKER_MODEL_NAME if session_state.reranker_active_for_session else "N/A",
            "k_initial": session_state.k_initial_retrieval,
            "rerank_thresh_default": session_state.reranker_score_threshold_session_default if session_state.reranker_active_for_session else "N/A",
            "base_sim_thresh_default": session_state.base_similarity_threshold_session_default if not session_state.reranker_active_for_session else "N/A",
            "embed_model": config.EMBEDDING_MODEL_NAME, "embed_device": rag_manager.device if rag_manager else "N/A",
            "db_path": os.path.abspath(config.DEFAULT_DB_DIR), "show_chunks": session_state.show_chunks_in_terminal,
            "total_turns": len(session_state.conversation_log)
        }
        full_log = {"session_summary": summary, "conversation_turns": session_state.conversation_log}
        os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
        print(f"\n[*] Saving log to: {log_file}...")
        with open(log_file, 'w', encoding='utf-8') as f: json.dump(full_log, f, indent=4, default=str) 
        print(f"[*] Log saved.")
    except Exception as e: print(f"\n[Error] Saving log: {e}"); traceback.print_exc()

# --- RAG Manager Class ---
class RAGManager:
    def __init__(self, config: ScriptConfig, session_state: SessionState):
        self.config = config; self.session_state = session_state; self.device = get_device_info()
        self.embedding_function = self._initialize_embeddings(); self.vector_db = self._initialize_db()
        self.reranker_instance: Optional[CrossEncoderReranker] = None
        if self.session_state.reranker_active_for_session: self._initialize_reranker() 
        if self.session_state.mode == "rag" and self.session_state.selected_llm_name: self._initialize_llm_and_chains()

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        print(f"[*] Embeddings: '{self.config.EMBEDDING_MODEL_NAME}' on '{self.device}'...")
        try:
            func = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME, model_kwargs={'device': self.device}, encode_kwargs={'normalize_embeddings': True}, show_progress=False)
            _ = func.embed_query("Test."); print("[*] Embeddings loaded."); return func
        except Exception as e: print(f"[Error] Embeddings init: {e}"); traceback.print_exc(); sys.exit(1)

    def _initialize_db(self) -> Chroma:
        print(f"[*] ChromaDB from: '{self.config.DEFAULT_DB_DIR}'")
        if not os.path.isdir(self.config.DEFAULT_DB_DIR): print(f"[Error] DB dir not found: {self.config.DEFAULT_DB_DIR}"); sys.exit(1)
        try:
            db = Chroma(persist_directory=self.config.DEFAULT_DB_DIR, embedding_function=self.embedding_function)
            print(f"[*] ChromaDB connected. Docs: {db._collection.count()}"); return db
        except Exception as e: print(f"[Error] DB init: {e}"); traceback.print_exc(); sys.exit(1)

    def _initialize_reranker(self):
        if not (RERANKING_COMPONENT_AVAILABLE and CrossEncoderReranker and HuggingFaceCrossEncoder):
            print("[Info] Re-ranking components N/A. Disabling for session."); self.session_state.reranker_active_for_session = False; self.reranker_instance = None; return
        print(f"[*] Reranker: '{self.config.DEFAULT_RERANKER_MODEL_NAME}' on '{self.device}'...")
        try:
            encoder = HuggingFaceCrossEncoder(model_name=self.config.DEFAULT_RERANKER_MODEL_NAME, model_kwargs={'device': self.device})
            self.reranker_instance = CrossEncoderReranker(model=encoder, top_n=self.session_state.k_initial_retrieval)
            print("[*] Reranker instance created.")
        except Exception as e: print(f"[Error] Reranker init: {e}"); traceback.print_exc(); print("[Warning] Proceeding without re-ranking."); self.session_state.reranker_active_for_session = False; self.reranker_instance = None
            
    def _initialize_llm_and_chains(self):
        if not self.session_state.selected_llm_name:
            print("[Warning] No LLM selected. Cannot init RAG chains."); self.session_state.llm_instance=None; self.session_state.rag_runnable=None; self.session_state.follow_up_runnable=None; return
        print(f"\n[*] Init Ollama LLM: '{self.session_state.selected_llm_name}', max_tokens={self.session_state.max_tokens}...")
        try:
            # REMOVED explicit host parameter. OllamaLLM will use OLLAMA_HOST env var or default.
            llm = OllamaLLM(model=self.session_state.selected_llm_name, num_predict=self.session_state.max_tokens)
            _ = llm.invoke("1+1?")
            print(f"[*] LLM initialized & tested."); self.session_state.llm_instance = llm
        except Exception as e:
            print(f"[Error] LLM init '{self.session_state.selected_llm_name}': {e}"); traceback.print_exc()
            self.session_state.llm_instance=None; self.session_state.rag_runnable=None; self.session_state.follow_up_runnable=None; return

        # This runnable expects 'docs_for_context' in the input dictionary, not 'question' for doc retrieval
        self.session_state.rag_runnable = (
            RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: self.format_docs_for_llm(x.get("docs_for_context", []))), 
                history=RunnableLambda(lambda x: self.session_state.memory.load_memory_variables({"question": x.get("question", "")}).get("history", ""))
            ) | self.session_state.rag_chain_template | self.session_state.llm_instance | StrOutputParser()
        )
        print("[*] Main RAG chain (re-)created.")
        self.session_state.follow_up_runnable = (
            RunnablePassthrough.assign(history=RunnableLambda(lambda x: self.session_state.memory.load_memory_variables({"question": x.get("question", "")}).get("history", ""))) |
            self.session_state.follow_up_chain_template | self.session_state.llm_instance | StrOutputParser()
        )
        print("[*] Follow-up RAG chain (re-)created.")

    def format_docs_for_llm(self, docs: List[Document]) -> str:
        if not docs: return "No relevant documents found."
        chunks = []
        for i, doc in enumerate(docs):
            meta = doc.metadata; content = doc.page_content
            score_info = f" (Score: {meta.get('final_score_used_for_filtering'):.4f})" if meta.get('final_score_used_for_filtering') is not None else ""
            title = meta.get('title', meta.get('vulnerabilityName', meta.get('pdf_title', meta.get('filename', 'N/A'))))
            src = meta.get('source_type', 'Unknown'); chunk_idx = meta.get('chunk_number', meta.get('chunk_index', '?'))
            id_parts = [f"CVE:{meta['cveID']}" if meta.get('cveID') else None, f"Video:{meta['video_id']}" if meta.get('video_id') else None]
            id_str = ", ".join(filter(None, id_parts)); id_info = f" ({id_str})" if id_str else ""
            chunks.append(f"--- Context Chunk {i+1} (Source: {src}{id_info}, Title: '{title}', Chunk: {chunk_idx}){score_info} ---\n{content}")
        return "\n\n".join(chunks)

    def get_final_documents_for_turn(self, query_text: str) -> Tuple[List[Document], str]:
        final_docs, info_str = [], "N/A"; k = self.session_state.k_initial_retrieval
        if self.session_state.reranker_active_for_session and self.reranker_instance and self.reranker_instance.model:
            retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
            initial_db_docs: List[Document] = []; actual_retrieved_count = 0
            try: initial_db_docs = retriever.invoke(query_text); actual_retrieved_count = len(initial_db_docs)
            except Exception as e: print(f"[Error] Initial retrieval: {e}"); traceback.print_exc()
            
            scored_filter_docs: List[Document] = []
            if initial_db_docs:
                pairs = [(query_text, d.page_content) for d in initial_db_docs]; scores_raw: Any = None
                try: 
                    if pairs: scores_raw = self.reranker_instance.model.score(pairs)
                except Exception as e: print(f"[Error] Reranker scoring: {e}"); traceback.print_exc()
                scores_list = []
                if scores_raw is not None:
                    if isinstance(scores_raw, np.ndarray): scores_list = scores_raw.tolist()
                    elif isinstance(scores_raw, list): scores_list = scores_raw
                    else: 
                        try: scores_list = list(scores_raw)
                        except TypeError: print(f"[Warning] Reranker scores type {type(scores_raw)} not list-convertible.")
                if len(scores_list) == actual_retrieved_count:
                    for i, od in enumerate(initial_db_docs):
                        nd = Document(page_content=od.page_content, metadata=od.metadata.copy())
                        try: nd.metadata['relevance_score'] = float(scores_list[i])
                        except (ValueError, TypeError): nd.metadata['relevance_score'] = None
                        nd.metadata['id_for_log'] = nd.metadata.get('id', f'db_idx_{i}'); scored_filter_docs.append(nd)
                else: 
                    print(f"[Warning] Score/doc mismatch ({len(scores_list)} vs {actual_retrieved_count}). No reranker scores applied."); 
                    scored_filter_docs = [Document(page_content=d.page_content, metadata=d.metadata.copy()) for d in initial_db_docs]
            
            q_thresh = self.session_state.reranker_score_threshold_session_default
            valid_scores = [d.metadata['relevance_score'] for d in scored_filter_docs if isinstance(d.metadata.get('relevance_score'), (float,int))]
            if valid_scores:
                min_s,max_s,avg_s = min(valid_scores),max(valid_scores),(sum(valid_scores)/len(valid_scores) if valid_scores else 0.0)
                print(f"\n--- Reranker Scores (Initial {actual_retrieved_count} docs) ---\nRange: {min_s:.4f} to {max_s:.4f} (Avg: {avg_s:.4f})")
                try:
                    inp = input(f"Threshold? (Default: {q_thresh:.2f}): ").strip()
                    if inp: q_thresh = float(inp); print(f"Interactive threshold: {q_thresh:.2f}")
                except ValueError: print("Invalid. Using session default.")
            elif scored_filter_docs: print("\n[Info] No valid reranker scores for stats/threshold.")
            else: print("\n[Info] No docs to rerank.")
            for d in scored_filter_docs:
                s = d.metadata.get('relevance_score')
                if isinstance(s,(float,int)) and s >= q_thresh: d.metadata['final_score_used_for_filtering']=s; final_docs.append(d)
            info_str = f"Reranked: k={k} ({actual_retrieved_count} DB), {len(scored_filter_docs)} scored, Thresh ({q_thresh:.2f}) -> {len(final_docs)} docs"
        else:
            sim_docs = self.vector_db.similarity_search_with_relevance_scores(query_text, k=k)
            actual_retrieved_count = len(sim_docs); q_thresh = self.session_state.base_similarity_threshold_session_default
            for d, s in sim_docs:
                if s >= q_thresh: d.metadata['similarity_score']=s; d.metadata['final_score_used_for_filtering']=s; final_docs.append(d)
            info_str = f"Direct: k={k}, {actual_retrieved_count} DB, Thresh ({q_thresh:.2f}) -> {len(final_docs)} docs"
        return final_docs, info_str

    def invoke_rag_chain_for_turn(self, query: str, docs_for_context: List[Document], is_follow_up: bool) -> str:
        if not self.session_state.llm_instance: return "Error: LLM not initialized."
        
        chain_to_use: Optional[Runnable] = None
        input_payload: Dict[str, Any] = {"question": query}

        if is_follow_up:
            chain_to_use = self.session_state.follow_up_runnable
            # For follow-up, context is derived from previously stored docs
            input_payload["context"] = self.format_docs_for_llm(self.session_state.last_retrieved_docs_for_follow_up or []) \
                                     if self.session_state.last_retrieved_docs_for_follow_up \
                                     else "No previous documents to use as context. Relying on chat history."
        else: # New query, use the rag_runnable which expects pre-fetched docs_for_context
            chain_to_use = self.session_state.rag_runnable
            input_payload["docs_for_context"] = docs_for_context # Pass retrieved docs

        if not chain_to_use: return "Error: RAG chain not available."

        print(f"\nAI ({self.session_state.selected_llm_name}): ", end="")
        resp_full = ""
        try:
            for chunk in chain_to_use.stream(input_payload): resp_full += chunk; sys.stdout.write(chunk); sys.stdout.flush()
            print()
        except Exception as e: print(f"\n[Error] RAG chain exec: {e}"); traceback.print_exc(); resp_full = f"(LLM Error: {type(e).__name__})"
        
        final_resp = resp_full.strip() or "(Empty LLM response)"
        self.session_state.memory.save_context({"question": query}, {"ai_response": final_resp})
        return final_resp
        
# --- Command Handler Class ---
class CommandHandler:
    def __init__(self, ss: SessionState, rm: RAGManager, cfg: ScriptConfig): self.state, self.manager, self.config = ss, rm, cfg
    def handle_command(self, inp: str) -> bool: 
        pts = inp.split(maxsplit=1); cmd, args = pts[0].lower(), (pts[1] if len(pts)>1 else None)
        if cmd=="/help": self._show_help()
        elif cmd=="/mode": self._toggle_mode()
        elif cmd=="/showchunks": self._toggle_show_chunks()
        elif cmd=="/set_tokens": self._set_tokens(args)
        elif cmd=="/set_rerank_threshold": self._set_rerank_thresh(args)
        elif cmd=="/set_base_threshold": self._set_base_sim_thresh(args)
        else: print(f"Unknown cmd: {cmd}. /help for options.")
        return True 
    def _show_help(self): print(f"\nCmds: /help, /mode, /showchunks, /set_tokens <N>, /set_rerank_threshold <f>, /set_base_threshold <0.0-1.0>, {', '.join(self.config.EXIT_KEYWORDS)}")
    def _toggle_mode(self):
        self.state.mode = "retrieval_only" if self.state.mode=="rag" else "rag"; print(f"Mode: {self.state.mode.replace('_',' ').title()}")
        if self.state.mode=="rag" and not self.state.llm_instance and self.state.selected_llm_name:
            print(f"Re-init LLM ({self.state.selected_llm_name})..."); self.manager._initialize_llm_and_chains()
        elif self.state.mode=="rag" and not self.state.selected_llm_name: print("No LLM selected for RAG.")
    def _toggle_show_chunks(self): self.state.show_chunks_in_terminal = not self.state.show_chunks_in_terminal; print(f"Show chunks: {'ON' if self.state.show_chunks_in_terminal else 'OFF'}")
    def _set_tokens(self, args: Optional[str]):
        if self.state.mode!="rag" or not self.state.llm_instance: print("RAG mode & LLM needed."); return
        try:
            val_s = args or input(f"Tokens (now: {self.state.max_tokens}, >50): ").strip()
            if not val_s: raise ValueError("No input.")
            new_v = int(val_s)
            if new_v>50: self.state.max_tokens=new_v; print(f"Tokens: {new_v}. Re-init LLM..."); self.manager._initialize_llm_and_chains()
            else: print("Must be > 50.")
        except (ValueError,TypeError) as e: print(f"Invalid. Usage: /set_tokens <num>. Error: {e}")
    def _set_rerank_thresh(self, args: Optional[str]):
        if not RERANKING_COMPONENT_AVAILABLE: print("Re-ranking N/A."); return
        try:
            val_s = args or input(f"Session Default Rerank Thresh (now: {self.state.reranker_score_threshold_session_default:.2f}): ").strip()
            if not val_s: raise ValueError("No input.")
            self.state.reranker_score_threshold_session_default = float(val_s)
            print(f"Session Default Rerank Thresh: {self.state.reranker_score_threshold_session_default:.2f}")
            if not self.state.reranker_active_for_session: print("Note: Re-ranking not active.")
        except (ValueError,TypeError) as e: print(f"Invalid. Usage: /set_rerank_threshold <f>. Error: {e}")
    def _set_base_sim_thresh(self, args: Optional[str]):
        try:
            val_s = args or input(f"Session Default Sim Thresh (now: {self.state.base_similarity_threshold_session_default:.2f}, 0-1): ").strip()
            if not val_s: raise ValueError("No input.")
            new_v = float(val_s)
            if 0.0<=new_v<=1.0: self.state.base_similarity_threshold_session_default=new_v; print(f"Session Default Sim Thresh: {new_v:.2f}"); (print("Note: Used when re-ranking OFF.") if self.state.reranker_active_for_session else None)
            else: print("Must be 0.0-1.0.")
        except (ValueError,TypeError) as e: print(f"Invalid. Usage: /set_base_threshold <f_0_to_1>. Error: {e}")

# --- Initial Setup Prompts ---
def initial_user_setup(ss: SessionState, cfg: ScriptConfig) -> bool:
    print("\n--- Initial Session Setup ---")
    mode_in = input(f"Mode (1.RAG, 2.RetrieveOnly) [Enter for {ss.mode}]: ").strip()
    ss.mode = {"1":"rag","2":"retrieval_only"}.get(mode_in, ss.mode); print(f"[*] Mode: {ss.mode.replace('_',' ').title()}")
    show_in = input(f"Show chunks? (Y/n) [Enter for {'Y' if ss.show_chunks_in_terminal else 'N'}]: ").strip().lower()
    ss.show_chunks_in_terminal = not(show_in=='n'); print(f"[*] Show chunks: {'ON' if ss.show_chunks_in_terminal else 'OFF'}")

    if ss.mode == "rag":
        print("\n--- Select LLM Model ---")
        if not cfg.AVAILABLE_MODELS: print("[Error] No LLMs in config."); return False
        def_llm = cfg.AVAILABLE_MODELS[0]; print("Available (from script config):")
        for i,n in enumerate(cfg.AVAILABLE_MODELS): print(f"  {i+1}. {n}")
        while True:
            llm_in = input(f"LLM (1-{len(cfg.AVAILABLE_MODELS)}) or name [Enter for '{def_llm}']: ").strip()
            tmp_n = def_llm if not llm_in else llm_in
            if llm_in.isdigit():
                try: tmp_n = cfg.AVAILABLE_MODELS[int(llm_in)-1]
                except (ValueError,IndexError): print("Invalid num."); continue
            if tmp_n and check_ollama_connection(tmp_n): ss.selected_llm_name = tmp_n; break
        print(f"[*] Selected LLM: {ss.selected_llm_name}")
        while True:
            tok_in = input(f"Max LLM tokens? [Enter for {ss.max_tokens}]: ").strip()
            if not tok_in: break
            try: v=int(tok_in); ss.max_tokens=v if v>50 else 50; break
            except ValueError: print("Invalid num.")
        print(f"[*] LLM max tokens: {ss.max_tokens}")

    ss.reranker_active_for_session = RERANKING_COMPONENT_AVAILABLE and not (input(f"Enable Re-ranking? (Y/n) [Enter for {'Y' if ss.reranker_active_for_session else 'N'}]: ").strip().lower() == 'n') if RERANKING_COMPONENT_AVAILABLE else False
    print(f"[*] Re-ranking: {'ON' if ss.reranker_active_for_session else 'OFF'}")
    
    if ss.reranker_active_for_session:
        def_k_re = cfg.DEFAULT_INITIAL_CANDIDATES_FOR_RERANKER
        while True: 
            k_in = input(f"Initial k for reranker? [Enter for {def_k_re}]: ").strip()
            if not k_in: ss.k_initial_retrieval = def_k_re; break
            try: v=int(k_in); ss.k_initial_retrieval=v if v>0 else def_k_re; break
            except ValueError: print("Invalid num.")
        print(f"[*] Initial k for reranker: {ss.k_initial_retrieval}")
        def_t_re = cfg.DEFAULT_RELEVANCE_SCORE_THRESHOLD_RERANKER
        while True: 
            t_in = input(f"Session Default Rerank Threshold? [Enter for {def_t_re:.2f}]: ").strip()
            if not t_in: ss.reranker_score_threshold_session_default = def_t_re; break
            try: ss.reranker_score_threshold_session_default = float(t_in); break
            except ValueError: print("Invalid float.")
        print(f"[*] Session Default Rerank Threshold: {ss.reranker_score_threshold_session_default:.2f}")
    else: 
        def_k_dir = cfg.DEFAULT_K_FOR_DIRECT_RETRIEVAL; ss.k_initial_retrieval = def_k_dir
        while True: 
            k_in = input(f"Max k for direct retrieval? [Enter for {ss.k_initial_retrieval}]: ").strip()
            if not k_in: break
            try: v=int(k_in); ss.k_initial_retrieval=v if v>0 else def_k_dir; break
            except ValueError: print("Invalid num.")
        print(f"[*] Max k for direct retrieval: {ss.k_initial_retrieval}")
        def_t_dir = cfg.DEFAULT_BASE_SIMILARITY_THRESHOLD
        while True: 
            t_in = input(f"Session Default Sim Threshold (0-1)? [Enter for {def_t_dir:.2f}]: ").strip()
            if not t_in: ss.base_similarity_threshold_session_default = def_t_dir; break
            try: v=float(t_in); ss.base_similarity_threshold_session_default=(v if 0.0<=v<=1.0 else def_t_dir); break
            except ValueError: print("Invalid float.")
        print(f"[*] Session Default Sim Threshold: {ss.base_similarity_threshold_session_default:.2f}")
    return True

# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(description="Refactored RAG Chat Application")
    parser.add_argument("--db_dir", default=None); parser.add_argument("--output_dir", default=None)
    cli_args = parser.parse_args()
    config = ScriptConfig(db_dir_override=cli_args.db_dir, output_dir_override=cli_args.output_dir)
    session = SessionState(config)
    if not initial_user_setup(session, config): sys.exit(1)
    rag_manager = RAGManager(config, session) 
    command_handler = CommandHandler(session, rag_manager, config)

    print("\n" + "="*30 + " Chat Session Started " + "="*30) 
    print(f"Mode: {session.mode.replace('_', ' ').title()}")
    if session.mode=="rag" and session.selected_llm_name: print(f"LLM: {session.selected_llm_name} | Max Tokens: {session.max_tokens}")
    if session.reranker_active_for_session: print(f"Rerank: ON (k={session.k_initial_retrieval}, Default Thresh >={session.reranker_score_threshold_session_default:.2f})")
    else: print(f"Rerank: OFF (Direct k={session.k_initial_retrieval}, Default Thresh >={session.base_similarity_threshold_session_default:.2f})")
    print(f"Show Chunks: {'ON' if session.show_chunks_in_terminal else 'OFF'}")
    print(f"'/help' or exit with: {', '.join(config.EXIT_KEYWORDS)}")
    print("=" * (len(" Chat Session Started ") + 60))

    try:
        while True:
            session.turn_count += 1; is_follow_up = False
            if session.turn_count > 1 and session.mode=="rag" and session.last_retrieved_docs_for_follow_up and session.llm_instance:
                if input("Follow-up? (y/n) [n]: ").strip().lower() == 'y': is_follow_up=True; print("[Using previous context]")
            
            user_q = input(f"\n[{session.turn_count}] You: ").strip()
            if not user_q: session.turn_count-=1; continue
            if user_q.lower() in config.EXIT_KEYWORDS: print("\nExiting..."); break
            if user_q.startswith('/'): command_handler.handle_command(user_q); session.turn_count-=1; continue

            final_docs_turn, retrieve_info, retrieve_log_status = [], "N/A", "N/A"
            
            if not is_follow_up: # For new queries, always retrieve
                print("Retrieving context...")
                try:
                    final_docs_turn, retrieve_info = rag_manager.get_final_documents_for_turn(user_q)
                    session.last_retrieved_docs_for_follow_up = final_docs_turn # Store for potential follow-up
                    retrieve_log_status = f"Success ({len(final_docs_turn)} docs)"
                except Exception as e: print(f"\n[Error] Retrieval failed: {e}"); traceback.print_exc(); retrieve_log_status = f"Error: {type(e).__name__}"
            else: # For follow-up, use previously stored docs
                final_docs_turn = session.last_retrieved_docs_for_follow_up or []
                retrieve_info = f"Using previous context ({len(final_docs_turn)} docs)"
                retrieve_log_status = "Success (Used Previous)" if final_docs_turn else "Failed (No Previous Context)"
            
            if session.show_chunks_in_terminal: display_retrieved_docs(user_q, final_docs_turn, retrieve_info)
            else: print(f"[*] Retrieved {len(final_docs_turn)} docs ({retrieve_info}, Display OFF).")

            ai_resp_log = "N/A (Retrieval Only or Error)"
            if session.mode == "rag":
                ai_resp_log = rag_manager.invoke_rag_chain_for_turn(user_q, final_docs_turn, is_follow_up) # Pass docs here

            log_docs_details = []
            if final_docs_turn:
                for i,d in enumerate(final_docs_turn):
                    meta = {k:(json.dumps(v) if isinstance(v,(dict,list,set,tuple)) else str(v) if not isinstance(v,(str,int,float,bool,type(None))) else v) for k,v in d.metadata.items()}
                    log_docs_details.append({"idx":i+1,"content":d.page_content,"metadata":meta})
            session.conversation_log.append({
                "turn":session.turn_count, "mode":session.mode, "is_follow_up":is_follow_up, "input":user_q, 
                "retrieval_status":retrieve_log_status, "retrieval_info":retrieve_info,
                "docs_count":len(log_docs_details), "docs_details":log_docs_details, "ai_response":ai_resp_log
            })
    except (EOFError,KeyboardInterrupt): print("\n\nUser interrupted.")
    except Exception as e: print(f"\n--- Error in main loop: {e} ---"); traceback.print_exc()
    finally: 
        print("\nSaving log...")
        save_chat_log(session, rag_manager, config) 
        print("\nChat session finished.")

if __name__ == "__main__":
    main()
