import os
import pandas as pd

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.data_cleaning import clean_and_merge_data
from src.preprocessing.feature_engineering import create_features
from src.faiss_index.embeddings_generator import generate_and_save_embeddings

def main():

    DATA_PATH = os.path.join(os.getcwd(), 'data')
    activity_df, sleep_df = load_raw_data(DATA_PATH)
    
    if activity_df is None or sleep_df is None:
        return
        
    cleaned_df = clean_and_merge_data(activity_df, sleep_df)
    final_df = create_features(cleaned_df)

    FAISS_SAVE_PATH = os.path.join(os.getcwd(), 'models', 'faiss_index')
    generate_and_save_embeddings(final_df, FAISS_SAVE_PATH)


if __name__ == "__main__":
    main()