# Base image with Python and Apache Spark
FROM bitnami/spark:latest

COPY . .

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint command to run scripts in order
CMD ["bash", "-c", "python3 src/predict.py"]
