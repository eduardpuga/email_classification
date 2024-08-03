from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sqlalchemy import create_engine

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Connect to the MySQL database
engine = create_engine('mysql+pymysql://root:root@db:3306/atc')

@app.route('/classify-email', methods=['POST'])
def classify_email():
    """
    Endpoint to classify an email.
    
    This endpoint accepts POST requests with a JSON body that should include:
    - client_id (int): The client ID.
    - email_body (str): The content of the email to be classified.
    
    Responses:
    - If the client has outstanding payments, returns a JSON with "success": False and a "reason".
    - If the client does not have outstanding payments, returns a JSON with "success": True and the "prediction".
    """
    # Get JSON data from the request
    data = request.get_json()
    client_id = data['client_id']
    email_body = data['email_body']
    
    # Check if the client has outstanding payments in the database
    impagos = pd.read_sql(f'SELECT * FROM impagos WHERE cliente_id = {client_id}', engine)
    if not impagos.empty:
        # If the client has outstanding payments, return a JSON response indicating the reason
        return jsonify({
            "success": False,
            "reason": "The client has outstanding payments"
        }), 200
    
    # If the client does not have outstanding payments, predict the email category
    prediction = model.predict([email_body])[0]
    
    # Return a JSON response with the prediction
    return jsonify({
        "success": True,
        "prediction": prediction
    }), 200

if __name__ == '__main__':
    """
    Entry point of the Flask application.
    
    If this file is run directly, the Flask application will start
    on host 0.0.0.0 and port 5000.
    """
    app.run(host='0.0.0.0', port=5000)