from pyspark.ml.regression import LinearSVRModel
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("HousePricePrediction").getOrCreate()

# Load model and data
model = LinearSVRModel.load("house_price_regression/model/svr_model")
data = spark.read.csv("house_price_regression/data/DATA-RUMAH-NEW.csv", header=True, inferSchema=True)

# Predict
predictions = model.transform(data)
predictions.show()
