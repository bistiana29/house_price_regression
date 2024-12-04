import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = df.drop(columns=['NO', 'NAMA RUMAH', 'HARGA'])
    y = df['HARGA']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if _name_ == "_main_":
    X_train, X_test, y_train, y_test = preprocess_data('/data/DATA-RUMAH.csv')
    print("Data preprocessing complete.")
