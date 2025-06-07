FROM confluentinc/cp-kafka:latest AS kafka

# Expose Kafka ports
EXPOSE 9092

# Set environment variables for Kafka configuration
ENV KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
ENV KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
ENV KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092
ENV KAFKA_BROKER_ID=1
ENV KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1

# Copy the topic creation script
COPY kafka-topic-create.sh /kafka-topic-create.sh
RUN chmod +x /kafka-topic-create.sh

# Start Kafka and then create topics
CMD [ "bash", "-c", "/etc/confluent/docker/run & sleep 20 && /kafka-topic-create.sh && wait" ]

# ---- Event Consumer Stage ----
FROM python:3.11-slim AS event-consumer
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-consumer.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY EventConsumer/consumer.py .
CMD ["python", "consumer.py"]

# ---- Anomaly Consumer Stage ----
FROM python:3.11-slim AS anomaly-consumer
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-consumer.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY AnomalyConsumer/consumer.py .
CMD ["python", "consumer.py"]

# ---- Migration CLI Stage ----
FROM python:3.11-slim AS migration-cli
WORKDIR /app
RUN pip install --no-cache-dir psycopg2-binary
COPY db_migrations ./migrations
COPY migrate.py .
ENTRYPOINT ["python", "migrate.py"]

# ---- Anomaly API Stage ----
FROM python:3.11-slim AS anomaly-api
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies
COPY AnomalyAPIService/requirements-anomaly-api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API service code
COPY AnomalyAPIService/main.py .

# Expose the port your API will run on (e.g., 8000)
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]