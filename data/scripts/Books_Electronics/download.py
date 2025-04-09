from datasets import load_dataset
import json
import os
from tqdm import tqdm

RAW_DATA_PATH = "../../raw"
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# Categorie da scaricare (salta quelle già presenti)
categories = {
    "Books": "raw_review_Books",
    "Electronics": "raw_review_Electronics"
}

meta_categories = {
    "Books": "raw_meta_Books",
    "Electronics": "raw_meta_Electronics"
}

def save_dataset(name, subset, fields, filename, split="full", max_records=None):
    dataset = load_dataset(name, subset, split=split, trust_remote_code=True, streaming=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for i, record in enumerate(tqdm(dataset, desc=f"📥 Salvando {filename}")):
            if max_records and i >= max_records:
                break
            filtered_record = {field: record.get(field, None) for field in fields}
            json.dump(filtered_record, f)
            f.write('\n')


if __name__ == "__main__":
    for cat, subset in categories.items():
        cat_path = os.path.join(RAW_DATA_PATH, cat)
        os.makedirs(cat_path, exist_ok=True)

        reviews_path = os.path.join(cat_path, "reviews.jsonl")
        if os.path.exists(reviews_path):
            print(f"⏩ Recensioni per {cat} già scaricate.")
        else:
            save_dataset(
                name="McAuley-Lab/Amazon-Reviews-2023",
                subset=subset,
                fields=["rating", "title", "text", "asin"],
                filename=reviews_path,
                max_records=50000
            )

    for cat, subset in meta_categories.items():
        cat_path = os.path.join(RAW_DATA_PATH, cat)
        os.makedirs(cat_path, exist_ok=True)

        meta_path = os.path.join(cat_path, "meta.jsonl")
        if os.path.exists(meta_path):
            print(f"⏩ Metadati per {cat} già scaricati.")
        else:
            save_dataset(
                name="McAuley-Lab/Amazon-Reviews-2023",
                subset=subset,
                fields=["main_category", "title", "parent_asin"],
                filename=meta_path,
                max_records=50000
            )
