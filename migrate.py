import os
import psycopg2

conn = psycopg2.connect(
    host=os.environ.get('POSTGRES_HOST', 'postgres'),
    port=os.environ.get('POSTGRES_PORT', 5432),
    dbname=os.environ.get('POSTGRES_DB', 'anomalydb'),
    user=os.environ.get('POSTGRES_USER', 'postgres'),
    password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
)
cur = conn.cursor()

with open('db_migrations/001_create_anomalies_table.sql', 'r') as f:
    sql = f.read()
    cur.execute(sql)
    conn.commit()

print("Migration applied successfully.")
cur.close()
conn.close()