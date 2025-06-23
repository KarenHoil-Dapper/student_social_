# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Dataset de ejemplo
data = pd.DataFrame({
    'edad': [25, 30, 45, 35],
    'ingreso': [50000, 60000, 80000, 75000],
    'compra': [0, 1, 1, 0]
})

X = data[['edad', 'ingreso']]
y = data['compra']

modelo = LogisticRegression()
modelo.fit(X, y)

# Guardar modelo
joblib.dump(modelo, 'model/modelo.pkl')
print("Modelo guardado en model/modelo.pkl")
