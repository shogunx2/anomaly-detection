import os
import psycopg2
import glob

conn = psycopg2.connect(
    host=os.environ.get('POSTGRES_HOST', 'postgres'),
    port=os.environ.get('POSTGRES_PORT', 5432),
    dbname=os.environ.get('POSTGRES_DB', 'anomalydb'),
    user=os.environ.get('POSTGRES_USER', 'postgres'),
    password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
)
cur = conn.cursor()

# Get all migration files and sort them
migration_files = sorted(glob.glob('db_migrations/*.sql'))

for migration_file in migration_files:
    print(f"Applying migration: {migration_file}")
    with open(migration_file, 'r') as f:
        sql = f.read()
        cur.execute(sql)
        conn.commit()
    print(f"Migration {migration_file} applied successfully.")

print("All migrations applied successfully.")
cur.close()
conn.close()
