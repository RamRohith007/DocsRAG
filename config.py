import os
from pathlib import Path
from typing import Dict, Any

# Document processing config
DOCUMENT_CONFIG = {
    "default_pdf": os.path.join(str(Path.home()), "Downloads", "got.pdf"),
    "documents_directory": os.path.join(str(Path.home()), "Desktop", "Learnings", "2025GEN", "Docs_RAG", "source_files"),
    "chunk_size": 1400,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "vectorstore_dir": "vectorstore"
}

# LLM config
LLM_CONFIG = {
    "model_name": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "model_type": "llama",  # TinyLlama uses same format as LLaMA
    "model_dir": os.path.join(str(Path.home()), "Desktop", "Learnings", "llm_models"),
    "temperature": 0.7,
    "max_new_tokens": 256,
    "context_length": 2048
}

# Retrieval config
RETRIEVAL_CONFIG = {
    "search_kwargs": {"k": 3},
    "chain_type": "stuff"
}

def get_config() -> Dict[str, Any]:
    """Return the complete configuration dictionary"""
    return {
        "document": DOCUMENT_CONFIG,
        "llm": LLM_CONFIG,
        "retrieval": RETRIEVAL_CONFIG
    }
