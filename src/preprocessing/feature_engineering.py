import pandas as pd
import numpy as np

def create_features(df):
    if df is None:
        return None

    df_featured = df.copy()

    df_featured['DayOfWeek'] = df_featured['Date'].dt.day_name()
    bins = [0, 7500, 10000, float('inf')]
    labels = ['Low', 'Moderate', 'High']
    df_featured['ActivityLevel'] = pd.cut(df_featured['Steps'], bins=bins, labels=labels, right=False) 
    conditions = [
        (df_featured['SleepHours'] < 7),
        (df_featured['SleepHours'] >= 7) & (df_featured['SleepHours'] <= 8),
        (df_featured['SleepHours'] > 8)
    ]
    outcomes = ['Poor', 'Optimal', 'Good']
    df_featured['SleepQuality'] = pd.Series(pd.NA, index=df_featured.index) 
    df_featured['SleepQuality'] = np.select(conditions, outcomes, default='Poor')

    return df_featured