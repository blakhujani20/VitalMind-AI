import faiss
import numpy as np
import os

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    if not index.is_trained:
        index.train(embeddings)

    index.add(embeddings)

    return index

def save_faiss_index(index, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    faiss.write_index(index, file_path)