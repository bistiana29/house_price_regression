import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['NO', 'NAMA RUMAH', 'HARGA'])
    y = df['HARGA']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Combine processed features and target into a DataFrame
    processed_data = pd.DataFrame(X_scaled, columns=X.columns)
    processed_data['HARGA'] = y

    # Save preprocessed data to the specified output path
    processed_data.to_csv(output_path, index=False)
    print(f"Data preprocessing complete. Saved to {output_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Define input and output paths
    input_path = './data/DATA-RUMAH.csv'
    output_path = './data/preprocessed-DATA-RUMAH.csv'

    # Call preprocess_data function
    X_train, X_test, y_train, y_test = preprocess_data(input_path, output_path)
