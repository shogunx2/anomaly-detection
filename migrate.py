import os
import psycopg2
import glob
import time
import sys

def wait_for_postgres(max_retries=10, retry_interval=3):
    """Wait for PostgreSQL to become available"""
    retries = 0
    while retries < max_retries:
        try:
            # Try to connect to PostgreSQL
            conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', 5432),
                user=os.environ.get('POSTGRES_USER', 'postgres'),
                password=os.environ.get('POSTGRES_PASSWORD', 'postgres'),
                # Connect to 'postgres' default DB first - our DB might not exist yet
                dbname='postgres'
            )
            conn.close()
            print("PostgreSQL is available!")
            return True
        except psycopg2.OperationalError as e:
            print(f"PostgreSQL not available yet, retrying in {retry_interval}s... ({retries+1}/{max_retries})")
            retries += 1
            time.sleep(retry_interval)
    
    print("Failed to connect to PostgreSQL after maximum retries")
    return False

def create_database_if_not_exists():
    """Create the application database if it doesn't exist"""
    db_name = os.environ.get('POSTGRES_DB', 'anomalydb')
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', 5432),
            user=os.environ.get('POSTGRES_USER', 'postgres'),
            password=os.environ.get('POSTGRES_PASSWORD', 'postgres'),
            dbname='postgres'
        )
        conn.autocommit = True  # Need autocommit for CREATE DATABASE
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        if not cur.fetchone():
            print(f"Creating database {db_name}...")
            cur.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully!")
        else:
            print(f"Database {db_name} already exists.")
        
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to create database: {e}")
        return False

def apply_migrations():
    """Apply all migration files"""
    db_name = os.environ.get('POSTGRES_DB', 'anomalydb')
    
    try:
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', 5432),
            dbname=db_name,
            user=os.environ.get('POSTGRES_USER', 'postgres'),
            password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
        )
        cur = conn.cursor()

        # Get all migration files and sort them
        migration_files = sorted(glob.glob('db_migrations/*.sql'))
        print(f"Found migration files: {migration_files}")

        for migration_file in migration_files:
            print(f"Applying migration: {migration_file}")
            try:
                with open(migration_file, 'r') as f:
                    sql = f.read()
                    cur.execute(sql)
                    conn.commit()
                print(f"Migration {migration_file} applied successfully.")
            except Exception as e:
                print(f"Error applying migration {migration_file}: {e}")
                conn.rollback()
                return False

        print("All migrations applied successfully.")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

# Main execution
if __name__ == "__main__":
    if not wait_for_postgres():
        sys.exit(1)
    
    if not create_database_if_not_exists():
        sys.exit(1)
    
    if not apply_migrations():
        sys.exit(1)
    
    print("Database setup completed successfully!")