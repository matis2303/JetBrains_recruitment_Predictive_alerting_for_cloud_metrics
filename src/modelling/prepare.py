from src.feature_engineering import statistical_features as sf
from src.preprocessing import preprocessing as al
import pandas as pd

def prepare_data(df: pd.DataFrame, 
                 value_column: str, 
                 target_column: str,
                 name: str, 
                 H: int, 
                 W: int,
                 threshold: float = 25.0,
                 labels_dict: dict=None) -> pd.DataFrame:
    

    if labels_dict is not None:
        df = al.add_labels_function_json(name, labels_dict, 'timestamp', df)
    else:
        df = al.add_labels_function(df, threshold, value_column, target_column)
    df = al.add_predictive_target(df, target_column, H)
    df = sf.add_lag(df, value_column, W)
    df = sf.add_mean_std_W(df, value_column, W)
    df = sf.add_relative_features(df, value_column)
    df = sf.add_advanced_features(df, value_column, W)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df = df.dropna().reset_index(drop=True).copy()
    return df