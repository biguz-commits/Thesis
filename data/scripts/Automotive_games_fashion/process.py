import json
import os
from tqdm import tqdm

RAW_DATA_PATH = "../../raw"
PROCESSED_DATA_PATH = "../../processed"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


def load_meta_data(meta_file):
    meta_dict = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"📄 Caricamento metadati da {os.path.basename(meta_file)}"):
            meta = json.loads(line)
            asin = meta.get('parent_asin')
            if asin:
                meta_dict[asin] = meta
    return meta_dict


def process_reviews(reviews_file, meta_dict, output_file):
    with open(reviews_file, 'r', encoding='utf-8') as rf, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line in tqdm(rf, desc=f"🔄 Processando {os.path.basename(reviews_file)}"):
            review = json.loads(line)
            asin = review.get('asin')
            meta = meta_dict.get(asin, {})
            processed = {
                "rating": review.get("rating"),
                "title": review.get("title"),
                "text": review.get("text"),
                "main_category": meta.get("main_category", "Unknown")
            }
            json.dump(processed, out_f)
            out_f.write('\n')


if __name__ == "__main__":
    categories = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]

    for cat in categories:
        meta_path = os.path.join(RAW_DATA_PATH, cat, "meta.jsonl")
        reviews_path = os.path.join(RAW_DATA_PATH, cat, "reviews.jsonl")
        output_path = os.path.join(PROCESSED_DATA_PATH, f"{cat}.jsonl")

        if not os.path.exists(meta_path) or not os.path.exists(reviews_path):
            print(f"⚠️  Dati mancanti per {cat}, salto...")
            continue

        print(f"\n🚀 Categoria: {cat}")
        meta_dict = load_meta_data(meta_path)
        process_reviews(reviews_path, meta_dict, output_path)

        print(f"✅ Salvato: {output_path}")
