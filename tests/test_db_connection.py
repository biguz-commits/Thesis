import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

class PostgresDB:
    def __init__(self):
        conn_params = {
            "user": os.getenv("POSTGRES_USER"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "database": os.getenv("POSTGRES_DB")
        }


        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **conn_params
        )

    def get_conn(self):
        return self.connection_pool.getconn()

    def put_conn(self, conn):
        self.connection_pool.putconn(conn)

    def close_all(self):
        self.connection_pool.closeall()

postgres_db = PostgresDB()

def test_connection():
    try:
        postgres_db = PostgresDB()
        conn = postgres_db.get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
            print("✅ Connessione riuscita!")
            print(f"📦 Versione del database: {db_version[0]}")
    except Exception as e:
        print("❌ Errore durante la connessione al database:")
        print(e)
    finally:
        if 'conn' in locals():
            postgres_db.put_conn(conn)

if __name__ == "__main__":
    test_connection()
