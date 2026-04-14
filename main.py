from src.data_loader import load_or_generate_data
from src.features import extract_features
from src.trainer import train_model, evaluate_model
import matplotlib.pyplot as plt
import os

def run_pipeline():
    # 1. Setup
    if not os.path.exists('outputs'): os.makedirs('outputs')
    if not os.path.exists('models'): os.makedirs('models')

    # 2. Data & Features
    raw_data = load_or_generate_data()
    processed_data = extract_features(raw_data)

    # 3. Split
    train_size = int(len(processed_data) * 0.8)
    train, test = processed_data[:train_size], processed_data[train_size:]
    
    features = ['hour', 'day_of_week', 'month', 'lag_24h']
    X_train, y_train = train[features], train['consumption']
    X_test, y_test = test[features], test['consumption']

    # 4. Train & Evaluate
    model = train_model(X_train, y_train)
    predictions, mse, r2 = evaluate_model(model, X_test, y_test)

    print(f"Pipeline Complete.\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")

    # 5. Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[-100:], label="Actual")
    plt.plot(predictions[-100:], label="Forecast", linestyle='--')
    plt.legend()
    plt.title("Energy Forecast - Last 100 Hours")
    plt.savefig('outputs/forecast_results.png')

if __name__ == "__main__":
    run_pipeline()