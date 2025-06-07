import os
from confluent_kafka import Consumer

print("Starting AnomalyConsumer...")

topic = os.environ.get("KAFKA_ANOMALY_TOPIC", "anomalies")
bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
print(f"Using topic: {topic}")

conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'AnomalyConsumer',
    'auto.offset.reset': 'latest'
}

print(f"Connecting to Kafka at {bootstrap_servers} for topic {topic}")

consumer = Consumer(conf)
consumer.subscribe([topic])

print(f"Consuming messages from topic: {topic} as AnomalyConsumer")
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        print(f"Received: {msg.value().decode('utf-8')}")
except KeyboardInterrupt:
    pass
finally:
    consumer.close()