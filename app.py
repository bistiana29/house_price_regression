from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

# Path ke file CSV
data_file = 'data/predict-DATA-RUMAH-NEW.csv/part-00000-2eaf8184-1451-42bc-aa68-8133c0621ec8-c000.csv'

# Membaca data dari CSV
data = pd.read_csv(data_file)

# Cek nama kolom
print(data.columns)  # Tambahkan ini untuk memeriksa nama kolom

@app.route('/')
def index():
    return render_template('index.html', results=data)

if __name__ == '__main__':
    app.run(debug=True)
