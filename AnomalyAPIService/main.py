from fastapi import FastAPI, HTTPException
import os
import psycopg2
from psycopg2.extras import RealDictCursor
print("Starting Anomaly API Service...")

app = FastAPI()
print("FastAPI app initialized.")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_conn():
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', 5432),
        dbname=os.environ.get('POSTGRES_DB', 'anomalydb'),
        user=os.environ.get('POSTGRES_USER', 'postgres'),
        password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
    )

@app.get("/anomalies/{enterprise_id}")
def list_anomalies(enterprise_id: str):
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    print(f"Fetching anomalies for enterprise_id: {enterprise_id}")
    print(f"SELECT * FROM anomalies WHERE enterprise_id = '{enterprise_id}'")
    cur.execute("SELECT * FROM anomalies WHERE enterprise_id = %s", (enterprise_id,))
    rows = cur.fetchall()
    print(f"Found {len(rows)} anomalies for enterprise_id: {enterprise_id}")
    cur.close()
    conn.close()
    return rows

@app.get("/anomaly/{anomaly_id}")
def get_anomaly(anomaly_id: int):
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM anomalies WHERE id = %s", (anomaly_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return row