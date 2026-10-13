"""
storage.py
----------
Manages data storage, index saving/loading, and folder structure.
"""

import os
import faiss
import pickle
from typing import List, Dict, Any

from utils.config import DATA_DIR, UPLOAD_DIR, INDEX_PATH, METADATA_PATH

def ensure_directories():
    """Ensure that the necessary data directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_index(dimension: int = 384) -> faiss.IndexFlatL2:
    """
    Load the FAISS index if it exists, otherwise create a new one.
    
    Args:
        dimension (int): The dimensionality of embeddings (default 384).
        
    Returns:
        faiss.IndexFlatL2: The loaded or newly created FAISS index.
    """
    ensure_directories()
    
    if os.path.exists(INDEX_PATH):
        try:
            return faiss.read_index(INDEX_PATH)
        except Exception:
            # Fallback if read fails
            return faiss.IndexFlatL2(dimension)
    else:
        return faiss.IndexFlatL2(dimension)

def save_index(index: faiss.Index) -> None:
    """
    Save the FAISS index to the configured path.
    
    Args:
        index (faiss.Index): The FAISS index to save.
    """
    ensure_directories()
    faiss.write_index(index, INDEX_PATH)

def load_metadata() -> List[Dict[str, Any]]:
    """
    Load the metadata list from a pickle file if it exists, 
    otherwise initialize an empty list.
    
    Returns:
        List[Dict[str, Any]]: The loaded metadata or an empty list.
    """
    ensure_directories()
    
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Fallback if read fails
            return []
    else:
        return []

def save_metadata(metadata: List[Dict[str, Any]]) -> None:
    """
    Save the metadata list to a pickle file.
    
    Args:
        metadata (List[Dict[str, Any]]): The metadata to save.
    """
    ensure_directories()
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
