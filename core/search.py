"""
search.py
---------
Handles the core search logic connecting the embedder and indexer.
"""

from typing import List, Dict

def search(query: str, embedder, indexer, top_k: int = 5) -> List[Dict]:
    """
    Perform a semantic search for a given query string.
    
    Args:
        query (str): The search query string.
        embedder: An instance of the Embedder class.
        indexer: An instance of the VectorIndexer class.
        top_k (int): The number of top results to return. Default is 5.
        
    Returns:
        List[Dict]: A list of matching results. Each result contains:
            - text (str): The chunk's content.
            - page (int): The source page number.
            - source (str): The originating document name.
            - score (float): The L2 distance score from FAISS.
    """
    if not query or not query.strip():
        return []
        
    # 1. Convert query to embedding
    # The embedder returns a 2D numpy array [1, dim]
    query_embeddings = embedder.generate_embeddings([query])
    
    # 2. Extract the single query embedding (1D array)
    query_vector = query_embeddings[0]
    
    # 3 & 4. Search the FAISS index (it already handles filtering inactive docs)
    results = indexer.search(query_vector, top_k=top_k)
    
    # 5. Return the list of results
    return results
