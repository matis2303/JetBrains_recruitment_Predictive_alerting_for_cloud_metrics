import pandas as pd
import numpy as np

def add_mean_std_W(df: pd.DataFrame,
             column: str,
             W: int
             ) -> pd.DataFrame:
    df[f"{column}_mean_W"] = df[column].rolling(window=W).mean()
    df[f"{column}_std_W"] = df[column].rolling(window=W).std()
    return df

def add_lag(df: pd.DataFrame, column: str, W: int) -> pd.DataFrame:
    if W <= 10:
        lags = range(1, W + 1)
    else:
        lags = set([1, 2, 3]) | set(range(5, W, 5)) | {W}
        lags = sorted(list(lags))
    
    for i in lags:
        df[f"{column}_lag_{i}"] = df[column].shift(i)
        
    return df


def add_relative_features(df: pd.DataFrame, column: str):

    df[f"{column}_diff_mean"] = df[column] - df[f"{column}_mean_W"]
    df[f"{column}_zscore"] = (df[column] - df[f"{column}_mean_W"]) / (df[f"{column}_std_W"] + 1e-9) #z-score calculation with no dividing by 0
    return df


def add_advanced_features(df: pd.DataFrame, column: str, W: int) -> pd.DataFrame:
    df[f"{column}_diff_1"] = df[column].diff(1)
    df[f"{column}_diff_3"] = df[column].diff(3)
    df[f"{column}_diff_5"] = df[column].diff(5)

    df[f"{column}_pct_change"] = df[column].pct_change().replace([np.inf, -np.inf], np.nan)

    df[f"{column}_min_W"] = df[column].rolling(window=W).min()
    df[f"{column}_max_W"] = df[column].rolling(window=W).max()

    df[f"{column}_range_W"] = df[f"{column}_max_W"] - df[f"{column}_min_W"]

    df[f"{column}_dist_max"] = df[column] - df[f"{column}_max_W"]
    df[f"{column}_dist_min"] = df[column] - df[f"{column}_min_W"]

    df[f"{column}_std_diff"] = df[f"{column}_std_W"].diff()

    return df


