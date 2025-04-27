from typing import Optional
from dataclasses import dataclass
from langchain_core.tools import tool
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from postgres.db_connection import PostgresDB


@dataclass
class QueryPurchasesInput:
    """Input parameters for querying the 'purchases' table."""
    start_date: Optional[str] = None  # Format 'YYYY-MM-DD'
    end_date: Optional[str] = None  # Format 'YYYY-MM-DD'


@tool
def query_purchases(input: QueryPurchasesInput) -> str:
    """
    Queries the 'purchases' table in the Postgres database and retrieves a list of past purchases.

    Args:
        input (QueryPurchasesInput): An object containing optional filters such as 'start_date' and 'end_date'.

    Returns:
        str: A formatted string listing the retrieved purchases, ready to be appended to a text prompt.

    Notes:
        - This tool does not filter by user ID.
        - If no purchases match the query, an empty list is returned.
    """
    postgres_db = PostgresDB()
    conn = postgres_db.get_conn()
    try:
        cursor = conn.cursor()
        query = "SELECT id, user_id, product_id, purchase_date, price, rating FROM purchases WHERE 1=1"
        params = []

        if input.start_date:
            query += " AND purchase_date >= %s"
            params.append(input.start_date)
        if input.end_date:
            query += " AND purchase_date <= %s"
            params.append(input.end_date)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return "No purchases found matching the given criteria."

        output = "List of retrieved purchases:\n\n"
        for row in rows:
            id, user_id, product_id, purchase_date, price, rating = row
            output += (
                f"- Purchase ID: {id}, User ID: {user_id}, "
                f"Product ID: {product_id}, Date: {purchase_date}, "
                f"Price: {price:.2f}€, Rating: {rating}\n"
            )

        return output.strip()

    finally:
        postgres_db.put_conn(conn)
