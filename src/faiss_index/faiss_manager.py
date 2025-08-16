import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "faiss_index")

def load_vector_store(folder_path=DEFAULT_INDEX_PATH, model_name='all-MiniLM-L6-v2'):
    if not os.path.exists(folder_path):
        return None

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
 
        vector_store = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
        
    except Exception as e:
        return None

def search_vector_store(vector_store, query_text, k=3):
    if vector_store is None:
        return []


    results = vector_store.similarity_search(query_text, k=k)
    return results
