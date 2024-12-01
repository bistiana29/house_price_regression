FROM python:3.12.3-slim

# Install dependencies
RUN apt-get update && apt-get install -y openjdk-11-jdk
RUN pip install pyspark pandas scikit-learn matplotlib

# Set work directory
WORKDIR /app

# Copy project
COPY . .

CMD ["bash"]
