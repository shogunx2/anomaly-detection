CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    source_event TEXT,
    resource_id TEXT,
    resource_name TEXT,
    severity TEXT,
    timestamp timestamp,
    enterprise_id TEXT,
    status VARCHAR
);