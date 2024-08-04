import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from sqlalchemy import create_engine
from utils import extract_numeric_features, extract_datetime_features
from sklearn.metrics import classification_report, accuracy_score



# Conectar a la base de datos MySQL
engine = create_engine('mysql+pymysql://root:root@db:3306/atc')

# Leer los datos de las tablas
emails = pd.read_sql('SELECT * FROM emails', engine)
impagos = pd.read_sql('SELECT * FROM impagos', engine)

# Quitar los emails de los clientes con impagos
clientes_con_impagos = impagos['client_id'].unique()
emails = emails[~emails['client_id'].isin(clientes_con_impagos)]

# Definir categorías (aquí ajusta las categorías según tu análisis previo)
emails['categoria'] = 'general'
emails.loc[emails['email'].str.contains('factura', case=False), 'categoria'] = 'factura'
emails.loc[emails['email'].str.contains('contrato', case=False), 'categoria'] = 'contrato'
emails.loc[emails['email'].str.contains('acceso', case=False), 'categoria'] = 'acceso'

# Crear el pipeline de clasificación
""" preprocessor = ColumnTransformer(
    transformers=[
        ('num', FunctionTransformer(extract_numeric_features, validate=False), ['client_id']),  # nombre de la columna
        ('date', FunctionTransformer(extract_datetime_features, validate=False), ['fecha_envio']),  # nombre de la columna
        ('text', TfidfVectorizer(), 'email')
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', MultinomialNB())
])
 """
# Crear el pipeline de clasificación usando solo el contenido del correo electrónico
pipeline = Pipeline([
    ('text', TfidfVectorizer()),  # Vectorizar el contenido del correo electrónico
    ('clf', MultinomialNB())      # Clasificador Naive Bayes
])

# Definir las características de entrada y la variable de destino
#X = emails[['client_id', 'fecha_envio', 'email']]
X = emails['email']
y = emails['categoria']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = pipeline.predict(X_test)

# Generar un informe de clasificación
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", report)
print("Accuracy:", accuracy)

# Guardar el modelo entrenado
joblib.dump(pipeline, '/app/model.pkl')  # Ajusta esta línea si es necesario
