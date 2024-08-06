import pandas as pd

# Extraer características de fecha
def extract_datetime_features(df):
    df['fecha_envio'] = pd.to_datetime(df['fecha_envio'])
    df['dia_semana'] = df['fecha_envio'].dt.dayofweek
    df['mes'] = df['fecha_envio'].dt.month
    df['es_finde'] = df['fecha_envio'].dt.dayofweek >= 5  # 5 y 6 son sábado y domingo
    return df

# Codificar las etiquetas de la variable categorica
def encode_labels(df, column_name):
    label_map = {'acceso': 0, 'contrato': 1, 'factura': 2, 'general': 3}
    df[column_name] = df[column_name].map(label_map)
    return df
# Decodificar la etiqueta codificada
def decode_label(encoded_label):
    label_map = {0: 'acceso', 1: 'contrato', 2: 'factura', 3: 'general'}
    return label_map[encoded_label]
