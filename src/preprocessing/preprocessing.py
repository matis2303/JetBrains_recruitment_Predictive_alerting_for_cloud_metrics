import pandas as pd


def add_labels_function(df: pd.DataFrame, threshold: float=25.0, value_column: str = 'value', target_column: str = 'is_incident') -> pd.DataFrame:
    df[target_column] = (df[value_column] >= threshold).astype(int)
    return df




def add_labels_function_json(file_name: str, 
               labels: dict,
               timestamp_columns: str,
               df_target: pd.DataFrame) -> pd.DataFrame:
    
    df_target[timestamp_columns] = pd.to_datetime(df_target[timestamp_columns])
    df_target["is_incident"] = 0


    incident_dates = labels.get(file_name, [])

    for interval in incident_dates:
        start = pd.to_datetime(interval[0])
        end = pd.to_datetime(interval[1])
        mask = (df_target[timestamp_columns] >= start) & (df_target[timestamp_columns] <= end)
        df_target.loc[mask, "is_incident"] = 1
    return df_target

def add_predictive_target(df: pd.DataFrame, target_name: str, H: int) -> pd.DataFrame:
    df = df.copy()
    df['target'] = df[target_name].rolling(window=H+1, min_periods=1).max().shift(-H) #i point + H points ahead
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    
    return df