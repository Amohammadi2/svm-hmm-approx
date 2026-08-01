import pandas as pd
import numpy as np


def load_gold18_history_data(file_path='../data-scraping/gold18_history.csv'):
    df = pd.read_csv(
        '../data-scraping/gold18_history.csv',
        converters={
            'Reopned': lambda x: np.nan if x == '-' else float(x.replace(',', '')),
            'Lowest': lambda x: np.nan if x == '-' else float(x.replace(',', '')),
            'Highest': lambda x: np.nan if x == '-' else float(x.replace(',', '')),
            'Final': lambda x: np.nan if x == '-' else float(x.replace(',', '')),
            'Change_Amount': lambda x: np.nan if x == '-' else float(x.replace(',', '')),
            'Change_Percent': lambda x: np.nan if x == '-' else float(x.replace('%', '')) / 100
        }
    )

    # Drop Jalali column
    df = df.drop('Date_Jalali', axis=1)

    df['Date_Miladi'] = pd.to_datetime(df['Date_Miladi'], format='%Y/%m/%d', errors='coerce')
    df.set_index('Date_Miladi', inplace=True)

    return df