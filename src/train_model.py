import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from preprocessing import preprocess_data

def train_model(input_path, model_path):
    # Preprocessing data
    processed_data_path = "./data/processed-DATA-RUMAH.csv"
    preprocess_data(input_path, processed_data_path)

    # Load preprocessed data
    data = pd.read_csv(processed_data_path)
    X = data.drop(columns=["HARGA"])
    y = data["HARGA"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse:.2f}")

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")
