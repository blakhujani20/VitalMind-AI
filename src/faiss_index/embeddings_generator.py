import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

def generate_and_save_embeddings(df, save_path, model_name='all-MiniLM-L6-v2'):
    if df is None:
        return

    sentences = []
    for index, row in df.iterrows():
        sentence = (
            f"On {row['DayOfWeek']}, the activity level was {row['ActivityLevel']} with {row['Steps']:.0f} steps and {row['Calories']:.0f} calories burned. "
            f"Sleep was {row['SleepQuality']} with {row['SleepHours']:.1f} hours."
        )
        sentences.append(sentence)
 
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(sentences, embedding=embedding_function)
    if save_path:
        print(f"Saving FAISS index to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        print("✅ FAISS index saved successfully.")
    
    return vectorstore