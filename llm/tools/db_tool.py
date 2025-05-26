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
    start_date: Optional[str] = None  
    end_date: Optional[str] = None  
    max_price: Optional[float] = None
    title_contains: Optional[str] = None


@tool
def query_purchases(input: QueryPurchasesInput) -> str:
    """
Executes a query on the 'purchases' table to retrieve information about completed transactions,
enriched with corresponding product details from the 'products' table.

For each purchase, the function returns a formatted output including:
- The price paid (`price`)
- The title of the purchased product (`title`)
- The average rating of the product (`average_rating`)
- The total number of ratings the product received (`rating_number`)

Optional filters can be applied using the following parameters:
- `start_date`: include only purchases made on or after this date ('YYYY-MM-DD')
- `end_date`: include only purchases made on or before this date ('YYYY-MM-DD')
- `max_price`: include only purchases where the price paid is less than or equal to this amount
- `title_contains`: include only products whose title contains the specified substring (case-insensitive)

Args:
    input (QueryPurchasesInput): An object containing the optional filtering criteria.

Returns:
    str: A formatted string listing the filtered purchases along with related product details.
         If no purchases are found, an informational message is returned.

Note:
    - Data is retrieved through a join between the `purchases` and `products` tables on the `product_id` key.
    - The `price` field refers to the price paid at the time of purchase (not the current product price).
    - The `average_rating`, `rating_number` fields are taken directly from the `products` table.
    - This tool does not apply any filter by user (`user_id`), so it returns purchases across all users.
"""

    postgres_db = PostgresDB()
    conn = postgres_db.get_conn()
    try:
        cursor = conn.cursor()
        query = """
            SELECT 
                p.price,
                pr.title,
                pr.average_rating,
                pr.rating_number
            FROM purchases p
            LEFT JOIN products pr ON p.product_id = pr.id
            WHERE 1=1
        """
        params = []

        if input.start_date:
            query += " AND p.purchase_date >= %s"
            params.append(input.start_date)
        if input.end_date:
            query += " AND p.purchase_date <= %s"
            params.append(input.end_date)

        if input.max_price is not None:
            query += " AND p.price <= %s"
            params.append(input.max_price)

        if input.title_contains:
            query += " AND pr.title ILIKE %s"
            params.append(f"%{input.title_contains}%")

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return "No purchases found matching the given criteria."

        output = "Purchases with product details:\n\n"
        for row in rows:
            price, title, avg_rating, rating_number = row
            output += (
                f"- Price: {price:.2f}€, "
                f"Title: {title}, "
                f"Average Rating: {avg_rating}, "
                f"Number of Ratings: {rating_number}\n"
            )

        return output.strip()

    finally:
        postgres_db.put_conn(conn)

