import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "faiss_index")

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(df, save_path=DEFAULT_SAVE_PATH):
    if df is None:
        return None, None

    sentences = []
    for _, row in df.iterrows():
        sentence = (
            f"On {row['DayOfWeek']}, activity level was {row['ActivityLevel']} with {row['Steps']} steps. "
            f"Resting heart rate was {row['RestingHeartRate']:.0f} bpm. "
            f"Sleep was {row['SleepQuality']} with {row['SleepHours']:.1f} hours."
        )
        sentences.append(sentence)


    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    vectorstore = FAISS.from_texts(sentences, embedding=embedding_function)

    vectorstore.save_local(save_path)
    return sentences, vectorstore



