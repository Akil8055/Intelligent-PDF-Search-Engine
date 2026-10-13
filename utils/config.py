import os

# Paths
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# Model
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Text Processing (WORD-BASED)
CHUNK_SIZE = 200
OVERLAP = 40

# Search
TOP_K = 5

# Performance
BATCH_SIZE = 32