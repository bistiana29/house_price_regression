from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel

# Inisialisasi Spark Session
spark = SparkSession.builder.appName("HousePricePrediction").getOrCreate()

# Muat model
model = LinearRegressionModel.load("model/linear_regression_model")

# Muat data baru
data = spark.read.csv("data/DATA-RUMAH-NEW.csv", header=True, inferSchema=True)

# Preprocessing data
assembler = VectorAssembler(
    inputCols=["LB", "LT", "KT", "KM", "GRS"],  # Kolom fitur
    outputCol="features"
)
data_transformed = assembler.transform(data)

# Prediksi harga rumah
predictions = model.transform(data_transformed)

# Pilih kolom relevan termasuk hasil prediksi
results = predictions.select("LB", "LT", "KT", "KM", "GRS", "prediction")

# Tampilkan hasil prediksi
results.show()

# Simpan hasil prediksi ke folder data
output_path = "data/predict-DATA-RUMAH-NEW.csv"
results.write.csv(output_path, header=True, mode="overwrite")
print(f"Hasil prediksi disimpan di: {output_path}")

# Hentikan Spark session
spark.stop()
