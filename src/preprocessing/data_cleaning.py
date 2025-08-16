import pandas as pd

def clean_and_merge_data(activity_df, sleep_df):
   
    if activity_df is None or sleep_df is None:
        return None

    activity_df = activity_df.rename(columns={
        'ActivityDate': 'Date', 'TotalSteps': 'Steps',
        'Calories': 'Calories'
    })
    sleep_df = sleep_df.rename(columns={
        'SleepDay': 'Date', 'TotalMinutesAsleep': 'TotalMinutesAsleep'
    })

    activity_df['Date'] = pd.to_datetime(activity_df['Date'], format='%m/%d/%Y')
    sleep_df['Date'] = pd.to_datetime(sleep_df['Date'], format='%m/%d/%Y %I:%M:%S %p').dt.date
    sleep_df['Date'] = pd.to_datetime(sleep_df['Date'])
    
    activity_df_subset = activity_df[['Id', 'Date', 'Steps', 'Calories']]
    sleep_df_subset = sleep_df[['Id', 'Date', 'TotalMinutesAsleep']]
    
    df = pd.merge(activity_df_subset, sleep_df_subset, on=['Id', 'Date'], how='left')

    df['SleepHours'] = df['TotalMinutesAsleep'] / 60
    df['SleepHours'] = df.groupby('Id')['SleepHours'].transform(lambda x: x.fillna(x.mean()))
    
    df.dropna(subset=['SleepHours'], inplace=True)
    
    df = df.drop(columns=['TotalMinutesAsleep'])
    
    return df