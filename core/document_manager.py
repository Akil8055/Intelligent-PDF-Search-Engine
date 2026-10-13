"""
document_manager.py
-------------------
Handles high-level document operations: adding, deleting, and listing documents.
"""

import os
import shutil
from typing import List

from .extractor import extract_text_with_pages
from .preprocessor import clean_pages
from .chunker import chunk_pages

from utils.config import UPLOAD_DIR

def add_document(file_path: str, embedder, indexer) -> None:
    """
    Process a PDF document and add it to the search index.
    
    Args:
        file_path (str): The path to the uploaded PDF file.
        embedder: An instance of Embedder.
        indexer: An instance of VectorIndexer.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(UPLOAD_DIR, file_name)
    
    # Store the file in the uploads directory
    if file_path != destination_path:
        shutil.copy2(file_path, destination_path)

    # 1. Extract text
    pages = extract_text_with_pages(destination_path)
    
    # 2. Clean text
    cleaned_pages = clean_pages(pages)
    
    # 3. Chunk text
    chunks = chunk_pages(cleaned_pages, source=file_name)
    
    # Ensure all chunks are marked as active
    for chunk in chunks:
        chunk["active"] = True
    
    # Extract just the text from chunks for the embedder
    texts_to_embed = [chunk["text"] for chunk in chunks]
    
    # 4. Generate embeddings
    embeddings = embedder.generate_embeddings(texts_to_embed)
    
    # 5. Add to indexer
    indexer.add_embeddings(embeddings, chunks)
    
    # 6. Save changes
    indexer.save()

def delete_document(file_name: str, indexer) -> bool:
    """
    Soft delete a document from the index.
    
    Args:
        file_name (str): The source file name to remove.
        indexer: An instance of VectorIndexer.
        
    Returns:
        bool: True if any chunks were marked inactive, False otherwise.
    """
    modified = False
    
    for chunk in indexer.metadata:
        if chunk.get("source") == file_name and chunk.get("active", True):
            chunk["active"] = False
            modified = True
            
    if modified:
        indexer.save()
        
    return modified

def list_documents(indexer) -> List[str]:
    """
    List all uniquely active documents in the index.
    
    Args:
        indexer: An instance of VectorIndexer.
        
    Returns:
        List[str]: A list of unique active document names.
    """
    active_docs = set()
    
    for chunk in indexer.metadata:
        if chunk.get("active", True) and "source" in chunk:
            active_docs.add(chunk["source"])
            
    return sorted(list(active_docs))
