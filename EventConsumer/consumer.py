import os
import json
import requests
from confluent_kafka import Consumer, Producer

print("Starting EventConsumer...")

topic = os.environ.get("KAFKA_TOPIC", "events")
bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ml_service_url = os.environ.get("ML_SERVICE_URL", "http://ml-service:8001")
print(f"Using topic: {topic}")
print(f"ML Service URL: {ml_service_url}")

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

def call_ml_service(event_data):
    """Call ML service to get anomaly prediction"""
    try:
        response = requests.post(
            f"{ml_service_url}/predict",
            json=event_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling ML service: {e}")
        # Return default result indicating no anomaly
        return {
            'is_anomaly': False,
            'error': str(e),
            'original_event': event_data,
            'reconstructed_event': event_data,
            'mismatched_cols': [],
            'n_mismatches': 0
        }

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
        
        try:
            event_data = json.loads(msg.value().decode('utf-8'))
            print(f"Loaded JSON: {event_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            continue
        
        if event_data.get("event_type") != "user_login_event":
            print(f"Skipping non-user_login_event: {event_data}")
            continue
        
        # Call ML service for anomaly detection
        print("Calling ML service for anomaly detection...")
        ml_result = call_ml_service(event_data)
        print(f"ML result: {ml_result}")
        
        # Only send to anomalies topic if ML service detected an anomaly
        if ml_result.get('is_anomaly', False):
            anomaly_event = {
                "event_type": "anomaly_detected",
                "data": event_data,
                "ml_result": ml_result
            }
            # Produce anomaly_event to Kafka
            producer.produce("anomalies", json.dumps(anomaly_event).encode('utf-8'))
            producer.flush()
            print(f"Produced anomaly_event to anomalies topic: {anomaly_event}")
        else:
            print("No anomaly detected, skipping Kafka production")
            
except KeyboardInterrupt:
    pass
finally:
    consumer.close() 