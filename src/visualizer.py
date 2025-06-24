import matplotlib.pyplot as plt
import seaborn as sns

def visualize_predictions(X_test, y_test, y_pred, df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test['Avg_Daily_Usage_Hours'], y=y_test, label='Real')
    sns.scatterplot(x=X_test['Avg_Daily_Usage_Hours'], y=y_pred, color='red', label='Predicción')
    plt.xlabel('Horas de Uso Diario')
    plt.ylabel('Mental Health Score')
    plt.title('Real vs Predicho según Horas de Uso Diario')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test['Avg_Daily_Usage_Hours'], y_test, c=X_test['Sleep_Hours_Per_Night'], cmap='viridis')
    plt.colorbar(scatter, label='Horas de Sueño por Noche')
    plt.xlabel('Horas de Uso Diario')
    plt.ylabel('Mental Health Score')
    plt.title('Mental Health Score vs Uso Diario, coloreado por Sueño')
    plt.show()

    sns.pairplot(df, x_vars=['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media'], 
                 y_vars='Mental_Health_Score', height=5, aspect=0.7, kind='reg')
    plt.suptitle('Relación entre variables y Mental Health Score', y=1.02)
    plt.show()
