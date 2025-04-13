import os
import json
from faker import Faker
import random
from tqdm import tqdm

# Percorsi
METADATA_DIR = "../products/metadata"
OUTPUT_DIR = "../products/sql_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup
fake = Faker()
USER_IDS = list(range(1, 6))  # Supponiamo 5 utenti già presenti

# Init per SQL
product_inserts = []
purchase_inserts = []
product_ids = set()

# Autoincrement fittizio per i prodotti
product_id_counter = 1
asin_to_product_id = {}

# Leggi tutti i metadata
for filename in os.listdir(METADATA_DIR):
    if not filename.endswith(".jsonl"):
        continue
    category = filename.replace("_metadata.jsonl", "")
    filepath = os.path.join(METADATA_DIR, filename)

    with open(filepath, "r") as f:
        for line in tqdm(f, desc=f"Processo {filename}"):
            try:
                rec = json.loads(line)
                title = rec.get("title")
                asin = rec.get("asin", fake.uuid4())  # Se manca asin, fake
                main_category = rec.get("main_category", category)
                avg_rating = rec.get("average_rating", 0.0)
                rating_number = rec.get("rating_number", 0)

                if asin not in asin_to_product_id:
                    product_id = product_id_counter
                    asin_to_product_id[asin] = product_id
                    product_id_counter += 1

                    safe_title = title.replace("'", "''")
                    product_inserts.append(
                        f"INSERT INTO products (id, asin, title, main_category, average_rating, rating_number) "
                        f"VALUES ({product_id}, '{asin}', '{safe_title}', '{main_category}', {avg_rating}, {rating_number});"
                    )

                # Simulazione acquisto (opzionale: puoi generarne + di 1 per prodotto)
                user_id = random.choice(USER_IDS)
                purchase_date = fake.date_between(start_date="-1y", end_date="today").strftime("%Y-%m-%d")
                price = round(random.uniform(5, 150), 2)
                rating = random.randint(3, 5)

                purchase_inserts.append(
                    f"INSERT INTO purchases (user_id, product_id, purchase_date, price, rating) "
                    f"VALUES ({user_id}, {asin_to_product_id[asin]}, '{purchase_date}', {price}, {rating});"
                )

            except Exception:
                continue

# Salvataggio SQL
with open(os.path.join(OUTPUT_DIR, "insert_products.sql"), "w") as f:
    f.write("\n".join(product_inserts))

with open(os.path.join(OUTPUT_DIR, "insert_purchases_from_metadata.sql"), "w") as f:
    f.write("\n".join(purchase_inserts))

# Output
"✅ File SQL generati: insert_products.sql e insert_purchases_from_metadata.sql"
