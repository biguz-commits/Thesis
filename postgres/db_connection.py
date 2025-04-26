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
