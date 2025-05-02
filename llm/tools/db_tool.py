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
        Esegue una query sulla tabella 'purchases' per ottenere informazioni sulle transazioni effettuate,
        arricchite con i dettagli dei prodotti corrispondenti dalla tabella 'products'.

        La funzione restituisce, per ciascun acquisto, un output formattato contenente:
        - Il prezzo pagato (`price`)
        - Il titolo del prodotto acquistato (`title`)
        - La valutazione media del prodotto (`average_rating`)
        - Il numero totale di valutazioni ricevute dal prodotto (`rating_number`)

        Filtri opzionali possono essere applicati tramite i parametri:
        - `start_date`: se specificata, vengono inclusi solo gli acquisti avvenuti dalla data in poi.
        - `end_date`: se specificata, vengono inclusi solo gli acquisti fino a quella data.

        Args:
            input (QueryPurchasesInput): Oggetto che può contenere i filtri temporali `start_date` e `end_date`
                                         in formato 'YYYY-MM-DD'.

        Returns:
            str: Una stringa formattata con l’elenco degli acquisti arricchiti dei dettagli prodotto.
                 Se non vengono trovati acquisti, viene restituito un messaggio informativo.

        Note:
            - I dati provengono da una join tra la tabella `purchases` e la tabella `products` sulla chiave `product_id`.
            - Il campo `price` si riferisce al prezzo pagato in ciascun acquisto (non al prezzo attuale del prodotto).
            - I campi `average_rating` e `rating_number` sono estratti direttamente dalla tabella `products`.
            - Questo tool non applica filtri sull'utente (`user_id`), quindi restituisce acquisti da tutti gli utenti.
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

