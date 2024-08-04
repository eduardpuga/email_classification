import pandas as pd

def extract_numeric_features(df):
    return df[['client_id']].values

def extract_datetime_features(df):
    return pd.to_datetime(df['fecha_envio']).astype('int64').values.reshape(-1, 1)