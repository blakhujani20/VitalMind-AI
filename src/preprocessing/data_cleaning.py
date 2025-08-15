import pandas as pd

def clean_health_data(df):
    if df is None:
        return None
    df_cleaned = df.copy()
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    if 'RestingHeartRate' in df_cleaned.columns and df_cleaned['RestingHeartRate'].isnull().any():
        median_hr = df_cleaned['RestingHeartRate'].median()
        df_cleaned['RestingHeartRate'] = df_cleaned['RestingHeartRate'].fillna(median_hr)
    return df_cleaned