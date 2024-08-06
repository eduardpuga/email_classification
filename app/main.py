from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sqlalchemy import create_engine
import datetime
from utils import extract_datetime_features, decode_label

app = Flask(__name__)

# Verificar que el modelo se cargar correctamente
try:
    pipeline = joblib.load('/app/model_data/model.pkl')
except Exception as e:
    raise ValueError(f"Fallo al cargar el modelo: {e}")

# Conectar a la base de datos MySQL
engine = create_engine('mysql+pymysql://root:root@db:3306/atc')

def is_valid_datetime(date_string):
    """
    Verificar si una cadena es una fecha y hora válida en formato 'YYYY-MM-DD HH:MM:SS' 
    """
    try:
        datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False


@app.route('/classify-email', methods=['POST'])
def classify_email():
    """
    Endpoint para clasificar un correo electrónico.

    Este endpoint acepta solicitudes POST con un cuerpo JSON que debe incluir:
    - client_id (int): El ID del cliente.
    - fecha_envio (str): La fecha y hora de envío del correo en formato 'YYYY-MM-DD HH:MM:SS'.
    - email_body (str): El contenido del correo electrónico a clasificar.

    Respuestas:
    - Si el cliente tiene impagos, devuelve un JSON con "exito": False y una "razon".
    - Si el cliente no tiene impagos, devuelve un JSON con "exito": True y la "prediccion".
    """
    print(request.headers)
    print(request.get_data())

    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415
    
    # Obtener los datos JSON de la solicitud
    data = request.get_json()

    # Validar client_id
    client_id = data.get('client_id')
    if not isinstance(client_id, int):
        return jsonify({"error": "client_id debe ser un integer"}), 400

    # Validar fecha_envio
    fecha_envio = data.get('fecha_envio')
    if not isinstance(fecha_envio, str) or not is_valid_datetime(fecha_envio):
        return jsonify({"error": "fecha_envio debe ser una cadena en formato YYYY-MM-DD HH:MM:SS"}), 400

    # Validar email_body
    email_body = data.get('email_body')
    if not isinstance(email_body, str):
        return jsonify({"error": "email_body debe ser un string"}), 400
    
    # Verificar si el cliente tiene impagos en la base de datos
    impagos = pd.read_sql(f'SELECT * FROM impagos WHERE client_id = {client_id}', engine)
    if not impagos.empty:
        # Si el cliente tiene impagos, devolver una respuesta JSON indicando la razón
        return jsonify({
            "exito": False,
            "razon": "El cliente tiene impagos"
        }), 200
    
    # Crear un DataFrame con la entrada del cliente
    input_data = pd.DataFrame({
        'client_id': [client_id],
        'fecha_envio': [fecha_envio],
        'email': [email_body]
    })
    
    # Extraer características de fecha
    input_data = extract_datetime_features(input_data)
    
    # Aplicar el pipeline de preprocesamiento y predecir la categoría del correo electrónico
    prediction = pipeline.predict(input_data)[0]
    prediction_label = decode_label(prediction)

    # Devolver una respuesta JSON con la predicción
    return jsonify({
        "exito": True,
        "prediccion": prediction_label
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
