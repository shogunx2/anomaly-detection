import os
from confluent_kafka import Consumer
import json
import psycopg2

print("Starting AnomalyConsumer...")
# Postgres connection details from environment
pg_host = os.environ.get('POSTGRES_HOST', 'postgres')
pg_port = os.environ.get('POSTGRES_PORT', 5432)
pg_db = os.environ.get('POSTGRES_DB', 'anomalydb')
pg_user = os.environ.get('POSTGRES_USER', 'postgres')
pg_password = os.environ.get('POSTGRES_PASSWORD', 'postgres')

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
        
        try:
            event_data = json.loads(msg.value().decode('utf-8'))
            # event data is in the key: data
            if 'data' in event_data:
                event_data = event_data['data']
            else:
                print("No 'data' key found in the message")
                continue
            print(f"Loaded JSON: {event_data}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            continue
        
        # Insert into anomalies table
        try:
            conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                dbname=pg_db,
                user=pg_user,
                password=pg_password
            )
            cur = conn.cursor()
            print("Connected to PostgreSQL database")
            cur.execute("""
                INSERT INTO anomalies (id, source_event, resource_id, resource_name, severity, timestamp, enterprise_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                event_data.get("id"),
                event_data.get("event_type"),
                event_data.get("resource_id"),
                event_data.get("resource_name"),
                'High',
                event_data.get("timestamp"),
                event_data.get("enterprise_id"),
                'Open'
            ))
            print("Executing insert statement")
            conn.commit()
            cur.close()
            conn.close()
            print(f"Inserted anomaly with id {event_data.get('id')}")
        except Exception as db_err:
            print(f"Failed to insert into anomalies table: {db_err}")
        
except KeyboardInterrupt:
    pass
finally:
    consumer.close()