import pandas as pd
import os

def load_raw_data(data_path, activity_file='dailyActivity_merged.csv', sleep_file='sleepDay_merged.csv'):

    try:
        activity_df = pd.read_csv(os.path.join(data_path, activity_file))
        sleep_df = pd.read_csv(os.path.join(data_path, sleep_file))
        return activity_df, sleep_df
    except FileNotFoundError as e:
        return None, None