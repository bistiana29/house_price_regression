FROM python:3.12-slim

RUN apt-get update && apt-get install -y openjdk-11-jdk-headless
RUN pip install pyspark pandas scikit-learn

WORKDIR /app
COPY . .

CMD ["python", "scripts/train_model.py"]
