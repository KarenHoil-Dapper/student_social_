from src.data_loader import load_data
from src.model_trainer import train_model
from src.visualizer import visualize_predictions
from src.predictor import predict_user_score

def main():
    df = load_data("data/Social_Bueno.xlsx")

    model, X_test, y_test, y_pred, mse, r2 = train_model(df)
    print(f"\n📊 Mean Squared Error: {mse:.2f}")
    print(f"📈 R-squared: {r2:.2f}")

    visualize_predictions(X_test, y_test, y_pred, df)

    while True:
        predict_user_score(model)
        cont = input("\n¿Quieres hacer otra predicción? (s/n): ").lower()
        if cont != 's':
            break

if __name__ == "__main__":
    main()
