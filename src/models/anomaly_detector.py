import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    if df is None:
        return None

    features = ['Steps', 'RestingHeartRate', 'SleepHours']
    df_features = df[features].copy()

    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(df_features)

    df['anomaly_score'] = model.predict(df_features)
    df['is_anomaly'] = df['anomaly_score'] == -1

    return df