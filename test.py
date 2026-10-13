from core.embedder import Embedder
from core.indexer import VectorIndexer

embedder = Embedder()
indexer = VectorIndexer()

texts = ["test document"]
embeddings = embedder.generate_embeddings(texts)

chunks = [{"text": "test document", "page": 1, "source": "test.pdf", "active": True}]

indexer.add_embeddings(embeddings, chunks)
indexer.save()