import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, accuracy_score
from utils import extract_datetime_features, encode_labels
import os

# Conectar a la base de datos MySQL
engine = create_engine('mysql+pymysql://root:root@db:3306/atc')

# Leer los datos de las tablas
emails = pd.read_sql('SELECT * FROM emails', engine)
impagos = pd.read_sql('SELECT * FROM impagos', engine)

# Quitar los emails de los clientes con impagos
clientes_con_impagos = impagos['client_id'].unique()
emails = emails[~emails['client_id'].isin(clientes_con_impagos)]

# Definición de categorías
emails['categoria'] = 'general'
emails.loc[emails['email'].str.contains('factura', case=False), 'categoria'] = 'factura'
emails.loc[emails['email'].str.contains('contrato', case=False), 'categoria'] = 'contrato'
emails.loc[emails['email'].str.contains('acceso', case=False), 'categoria'] = 'acceso'

# Extraer características de fecha
emails = extract_datetime_features(emails)

# Crear el pipeline de clasificación con todos los datos
preprocessor = ColumnTransformer(
    transformers=[
        ('client', OneHotEncoder(handle_unknown='ignore'), ['client_id']),
        ('date', OneHotEncoder(handle_unknown='ignore'), ['dia_semana', 'mes', 'es_finde']),
        ('text', TfidfVectorizer(), 'email')
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier())
])

# Crear el LabelEncoder para la variable objetivo
emails = encode_labels(emails, 'categoria')

# Definir las características de entrada y la variable de destino
X = emails[['client_id', 'dia_semana', 'mes', 'es_finde', 'email']]
y = emails['categoria']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = pipeline.predict(X_test)

# Generar un informe de clasificación
report = classification_report(y_test, y_pred, zero_division=1)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", report)
print("Accuracy:", accuracy)

# Guardar el modelo entrenado
model_path = '/train/model_data/model.pkl'
joblib.dump(pipeline, model_path)

# Verificar si el modelo es guardado correctamente
if not os.path.exists(model_path):
    raise ValueError(f"Modelo no guardado correctamente en {model_path}")