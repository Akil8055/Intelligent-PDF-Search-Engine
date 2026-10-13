"""
indexer.py
----------
Manages the FAISS vector database for storing embeddings and performing
fast similarity search.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict

from utils.config import INDEX_PATH, METADATA_PATH
from utils.storage import ensure_directories


class VectorIndexer:
    def __init__(self, embedding_dim: int = 384, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        """
        Initialize the FAISS index for vector storage and search.
        
        Args:
            embedding_dim (int): Dimensionality of the embeddings (default 384 for all-MiniLM-L6-v2).
            index_path (str): File path to save/load the FAISS index.
            metadata_path (str): File path to save/load the chunk metadata.
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize an empty L2 distance index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata: List[Dict] = []
        
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Add new embeddings to the index and their corresponding chunks to metadata.
        
        Args:
            embeddings (np.ndarray): 2D numpy array of embeddings.
            chunks (List[Dict]): List of chunk dicts containing text, page, source, active.
                                 Must be the same length as embeddings.
        """
        if embeddings is None or len(embeddings) == 0:
            return
            
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
            
        # FAISS expects float32 numpy arrays
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Store metadata directly (no embeddings in metadata)
        self.metadata.extend(chunks)
            
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search the index for the most similar chunks to the query embedding.
        Skips inactive documents.
        
        Args:
            query_embedding (np.ndarray): 1D or 2D array of the query vector.
            top_k (int): Number of top active results to return.
            
        Returns:
            List[Dict]: Top-k chunks with their metadata and distance scores.
        """
        if self.index.ntotal == 0:
            return []
            
        # Ensure query is a 2D float32 array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Fetch more than top_k initially to account for filtered out inactive documents
        search_k = min(self.index.ntotal, max(top_k * 5, 20)) 
        
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                # Skip inactive documents
                if not self.metadata[idx].get("active", True):
                    continue
                    
                result = self.metadata[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
                
                # Stop if we have enough active results
                if len(results) >= top_k:
                    break
                    
        return results
        
    def save(self) -> None:
        """Save the FAISS index and metadata to disk."""
        ensure_directories()
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk if they exist.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return True
            except Exception:
                return False
        return False
