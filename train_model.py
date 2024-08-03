import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Conectar a la base de datos MySQL
engine = create_engine('mysql+pymysql://root:root@localhost:3306/atc')

# Leer las tablas
emails = pd.read_sql('SELECT * FROM correos', engine)
impagos = pd.read_sql('SELECT * FROM impagos', engine)

# Filtrar correos de clientes impagos
impagos_ids = impagos['cliente_id'].tolist()
emails = emails[~emails['cliente_id'].isin(impagos_ids)]

# Preparar los datos
X = emails['contenido']  # Asumiendo que la columna de texto se llama 'contenido'
y = emails['categoria']  # Asumiendo que tienes una columna de categorías

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el pipeline de clasificación
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(pipeline, 'app/model.pkl')
