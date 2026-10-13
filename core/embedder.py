"""
embedder.py
-----------
Handles the generation of dense vector embeddings for text using
Sentence Transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): HuggingFace model name for SentenceTransformers.
        """
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate normalized embeddings for a list of text strings.
        
        Args:
            texts (List[str]): List of texts to embed.
            batch_size (int): Batch size for inference. Default is 32.
            
        Returns:
            np.ndarray: A 2D numpy array of normalized embeddings.
        """
        if not texts:
            return np.array([])

        # Generate embeddings
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        
        # Normalize embeddings using L2 norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        normalized_embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
        
        return normalized_embeddings

# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    embedder = Embedder()
    sample_texts = ["Hello world", "Sentence Transformers are great."]
    embeddings = embedder.generate_embeddings(sample_texts)
    print(f"Generated normalized embeddings of shape: {embeddings.shape}")
