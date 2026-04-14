import pandas as pd
def extract_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Lag feature: The consumption 24 hours ago
    df['lag_24h'] = df['consumption'].shift(24)
    return df.dropna()