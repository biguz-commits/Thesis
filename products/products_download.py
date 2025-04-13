import os
import json
from tqdm import tqdm
from datasets import load_dataset

# Cartella di salvataggio
SAVE_DIR = "metadata"
os.makedirs(SAVE_DIR, exist_ok=True)

# Categorie da scaricare
categories = {
    "All_Beauty": "raw_meta_All_Beauty",
    "Amazon_Fashion": "raw_meta_Amazon_Fashion",
    "Automotive": "raw_meta_Automotive",
    "Books": "raw_meta_Books",
    "Electronics": "raw_meta_Electronics",
    "Video_Games": "raw_meta_Video_Games",
}

# Funzione per salvare i dati in jsonl
def save_metadata_jsonl(category, subset_name, max_records=50000):
    print(f"Scaricando dati per categoria: {category}")
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", subset_name, split="full", trust_remote_code=True, streaming=True)

    filename = os.path.join(SAVE_DIR, f"{category}_metadata.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        for i, record in enumerate(tqdm(dataset, desc=f"Scrittura {filename}")):
            if i >= max_records:
                break
            filtered = {
                "main_category": record.get("main_category"),
                "title": record.get("title"),
                "average_rating": record.get("average_rating"),
                "rating_number": record.get("rating_number"),
            }
            json.dump(filtered, f)
            f.write("\n")

# Scarica per ogni categoria
for cat, subset in categories.items():
    save_metadata_jsonl(cat, subset)
