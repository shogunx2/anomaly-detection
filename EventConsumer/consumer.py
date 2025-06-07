import os
from confluent_kafka import Consumer, Producer

print("Starting EventConsumer...")

topic = os.environ.get("KAFKA_TOPIC", "events")
bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
print(f"Using topic: {topic}")

conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'EventConsumer',
    'auto.offset.reset': 'latest'
}

print(f"Connecting to Kafka at {bootstrap_servers} for topic {topic}")

consumer = Consumer(conf)
consumer.subscribe([topic])

# Set up a Kafka producer
producer = Producer({'bootstrap.servers': bootstrap_servers})

print(f"Consuming messages from topic: {topic} as EventConsumer")
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        print(f"Received: {msg.value().decode('utf-8')}")
        import json
        try:
            event_data = json.loads(msg.value().decode('utf-8'))
            print(f"Loaded JSON: {event_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
        
        if event_data.get("event_type") != "user_login_event":
            print(f"Skipping non-user_login_event: {event_data}")
            continue
        # Here you would typically process the event_data
        
        anomaly_event = {
            "event_type": "anomaly_detected",
            "data": event_data
        }
        # Produce anomaly_event to Kafka (to the same topic or another topic as needed)
        producer.produce("anomalies", json.dumps(anomaly_event).encode('utf-8'))
        producer.flush()
        print(f"Produced anomaly_event to topic {topic}: {anomaly_event}")
except KeyboardInterrupt:
    pass
finally:
    consumer.close()