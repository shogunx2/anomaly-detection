#!/bin/bash

# Wait for Kafka to be ready
sleep 60

# Create 'events' topic
kafka-topics --create --if-not-exists --topic events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Create 'anomalies' topic
kafka-topics --create --if-not-exists --topic anomalies --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Cross-check if topics are created
echo "Checking if topics 'events' and 'anomalies' exist..."
kafka-topics --list --bootstrap-server localhost:9092 | grep -E '^(events|anomalies)$'

if kafka-topics --list --bootstrap-server localhost:9092 | grep -q '^events$' && \
   kafka-topics --list --bootstrap-server localhost:9092 | grep -q '^anomalies$'; then
    echo "Both topics 'events' and 'anomalies' are created successfully."
else
    echo "Error: One or both topics were not created."
    exit 1
fi