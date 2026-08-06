import pandas as pd
import numpy as np


def calculate_tgju_log_returns(df: pd.DataFrame):
    """ Calculates log returns for data coming from the TGJU scraper
    This methods assumes there is a 'Final' column in string format "196,546,865"
    """
    df = df.copy()
    df['Final_F'] = df['Final'].apply(lambda x: np.nan if x == '-' else float(x.replace(',', '')))
    df["Log_Return"] = np.log(df["Final_F"] / df["Final_F"].shift(1)) * 100

    return df