import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['NO', 'NAMA RUMAH', 'HARGA'])
    y = df['HARGA']

    # standarisasi
    X = sc_X.fit_transform(X)
    y = np.log1p(y)

    # save preprocessed data
    processed_data = pd.DataFrame(X, columns=df.drop(columns=['NO', 'NAMA RUMAH', 'HARGA']).columns)
    processed_data['HARGA'] = y

    pd.DataFrame(processed_data).to_csv(output_path, index=False)

if _name_ == '_main_':
    preprocess_data('data/DATA-RUMAH.csv', 'data/processed-DATA-RUMAH.csv')
