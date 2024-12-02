from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("HousePriceSVR").getOrCreate()

# Load df
df = spark.read.csv("data/processed-DATA-RUMAH.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("_c0", "label")

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2])

# Train SVR model
from pyspark.ml.regression import LinearSVR
svr = LinearSVR(maxIter=100)
model = svr.fit(train_data)

# Save model
model.write().overwrite().save("/model/svr_model")
