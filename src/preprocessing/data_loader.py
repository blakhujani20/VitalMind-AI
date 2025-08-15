import pandas as pd
import os
from data_cleaning import clean_health_data
from feature_engineering import create_features

def load_fitbit_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None