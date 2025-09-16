from langchain_community.llms import CTransformers
from config import LLM_CONFIG
import os
from typing import Optional

def get_llm(
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    model_dir: Optional[str] = None,
    **kwargs
) -> CTransformers:
    """Initialize and return a configured CTransformers LLM"""
    config = {
        **LLM_CONFIG,
        **kwargs
    }
    if model_name:
        config["model_name"] = model_name
    if model_type:
        config["model_type"] = model_type
    if model_dir:
        config["model_dir"] = model_dir
        
    model_path = os.path.join(config["model_dir"], config["model_name"])
    
    return CTransformers(
        model=model_path,
        model_type=config["model_type"],
        config={
            'temperature': config["temperature"],
            'max_new_tokens': config["max_new_tokens"],
            'context_length': config["context_length"]
        }
    )

# Initialize LLM with default parameters
llm = get_llm()
