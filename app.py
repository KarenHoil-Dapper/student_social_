# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
modelo = joblib.load('model/modelo.pkl')

@app.route('/')
def home():
    return "API de Predicción de Compras"

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.get_json()
    df = pd.DataFrame([datos])
    pred = modelo.predict(df)
    return jsonify({'resultado': int(pred[0])})

if __name__ == '__main__':
    app.run(debug=True)
