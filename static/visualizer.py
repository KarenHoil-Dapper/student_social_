import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_user_score_on_distribution(data_file, user_score, output_file="static/user_mh_score.png"):
    """
    Genera un histograma de la puntuación de salud mental de todos los usuarios,
    resaltando la puntuación del usuario actual.
    """
    df = pd.read_excel(data_file)
    scores = df["Mental_Health_Score"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=10, color="#667eea", edgecolor="black", alpha=0.7, label="Usuarios")
    plt.axvline(user_score, color="red", linestyle="--", linewidth=2, label=f"Tu puntuación: {user_score:.2f}")
    plt.title("Distribución de Salud Mental (Usuarios)")
    plt.xlabel("Puntuación de Salud Mental")
    plt.ylabel("Cantidad de Usuarios")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()