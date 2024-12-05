from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel

# Inisialisasi Spark Session
spark = SparkSession.builder.appName("HousePricePrediction").getOrCreate()

# Muat model
model = LinearRegressionModel.load("house_price_regression/model/linear_regression_model")

# Muat data baru
data = spark.read.csv("house_price_regression/data/DATA-RUMAH-NEW.csv", header=True, inferSchema=True)

# Preprocessing data
assembler = VectorAssembler(
    inputCols=["LB", "LT", "KT", "KM", "GRS"],  # Kolom fitur
    outputCol="HARGA"
)
data_transformed = assembler.transform(data)

# Prediksi harga rumah
predictions = model.transform(data_transformed)

# Tampilkan hasil prediksi
predictions.select("LB", "LT", "KT", "KM", "GRS").show()
