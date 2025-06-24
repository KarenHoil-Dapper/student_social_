import numpy as np

def predict_user_score(model):
    try:
        usage = float(input("Promedio de horas diarias de uso: "))
        sleep = float(input("Horas de sueño por noche: "))
        conflicts = int(input("Número de conflictos por redes sociales: "))

        input_data = np.array([[usage, sleep, conflicts]])
        prediction = model.predict(input_data)

        print(f"\nPredicción de Mental Health Score: {prediction[0]:.2f}")
    except ValueError:
        print("❌ Entrada inválida. Intenta de nuevo.")
