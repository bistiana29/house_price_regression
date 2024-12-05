from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

def train_model(input_path, model_path):
    spark = SparkSession.builder.appName("HousePricePrediction").getOrCreate()
    data = spark.read.csv(input_path, header=True, inferSchema=True)

    # Menentukan kolom fitur dan kolom target
    feature_columns = ["LB", "LT", "KT", "KM", "GRS"]  # Ganti dengan kolom fitur yang sesuai
    target_column = "HARGA"

    # Menggunakan VectorAssembler untuk menggabungkan kolom fitur menjadi satu kolom vektor
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_transformed = assembler.transform(data)

    # Membagi data menjadi train dan test
    train_data, test_data = data_transformed.randomSplit([0.8, 0.2], seed=42)

    # Melatih model regresi linier
    lr = LinearRegression(featuresCol="features", labelCol=target_column)
    model = lr.fit(train_data)
    print("Model training completed.")

    # Evaluasi model
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)
    print(f"Mean Squared Error on test data: {mse:.2f}")

    # Simpan model
    model.save(model_path)
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    input_path = "data/preprocessed-DATA-RUMAH.csv"
    model_path = "model/linear_regression_model"
    train_model(input_path, model_path)
