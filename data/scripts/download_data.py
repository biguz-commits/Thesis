from datasets import load_dataset
import json
import os
from tqdm import tqdm

RAW_DATA_PATH = "../raw"
os.makedirs(RAW_DATA_PATH, exist_ok=True)

categories = {
    "All_Beauty": "raw_review_All_Beauty",
}

meta_categories = {
    "All_Beauty": "raw_meta_All_Beauty",
}


def save_dataset(name, subset, fields, filename, split="full", max_records=None):
    dataset = load_dataset(name, subset, split=split, trust_remote_code=True, streaming=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for i, record in enumerate(tqdm(dataset, desc=f"Saving {filename}")):
            if max_records and i >= max_records:
                break
            filtered_record = {field: record.get(field, None) for field in fields}
            json.dump(filtered_record, f)
            f.write('\n')


if __name__ == "__main__":
    for cat, subset in categories.items():
        cat_path = os.path.join(RAW_DATA_PATH, cat)
        os.makedirs(cat_path, exist_ok=True)

        save_dataset(
            name="McAuley-Lab/Amazon-Reviews-2023",
            subset=subset,
            fields=["rating", "title", "text", "asin"],
            filename=os.path.join(cat_path, "reviews.jsonl"),
            max_records=50000
        )

    for cat, subset in meta_categories.items():
        cat_path = os.path.join(RAW_DATA_PATH, cat)
        os.makedirs(cat_path, exist_ok=True)

        save_dataset(
            name="McAuley-Lab/Amazon-Reviews-2023",
            subset=subset,
            fields=["main_category", "title", "parent_asin"],
            filename=os.path.join(cat_path, "meta.jsonl"),
            max_records=50000
        )
