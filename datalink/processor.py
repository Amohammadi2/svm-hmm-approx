import pandas as pd
import numpy as np


def calculate_tgju_log_returns(df: pd.DataFrame):
    """ Calculates log returns for data coming from the TGJU scraper
    This methods assumes there is a 'Final' column in string format "196,546,865"
    """
    res = pd.DataFrame()
    df = df.copy().iloc[::-1]
    res['Date'] = df['Date_Miladi']
    res['Final_F'] = df['Final'].apply(lambda x: np.nan if x == '-' else float(x.replace(',', '')))
    res["Log_Return"] = np.log(res["Final_F"] / res["Final_F"].shift(1)) * 100
    res = res.reset_index(drop=True)

    return res