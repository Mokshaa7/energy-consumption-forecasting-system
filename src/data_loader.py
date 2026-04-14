import pandas as pd
import numpy as np
import os

def load_or_generate_data(filepath='data/energy_data.csv'):
    if not os.path.exists('data'): os.makedirs('data')
    
    print("Fetching dataset...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='h')
    
    # Simulate realistic patterns
    hour_effect = 10 * np.sin(2 * np.pi * dates.hour / 24)
    daily_noise = np.random.normal(0, 2, len(dates))
    energy = 50 + hour_effect + daily_noise
    
    df = pd.DataFrame({'timestamp': dates, 'consumption': energy})
    df.to_csv(filepath, index=False)
    return df