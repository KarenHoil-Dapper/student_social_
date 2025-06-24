from flask import Flask, request, jsonify
from train_model import ExcelSocialMediaPredictor

app = Flask(__name__)

# Cargar modelo pre-entrenado
predictor = ExcelSocialMediaPredictor()
predictor.load_trained_model('mi_modelo_redes_sociales.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir datos del formulario
        user_data = request.json
        
        # Hacer predicción
        predictions = predictor.predict_new_user(user_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)