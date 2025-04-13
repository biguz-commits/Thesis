import os
import json
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_store")
    collection = client.get_or_create_collection("amazon-reviews")

    file_path = os.path.join("..", "data", "processed", "Electronics.jsonl")
    category = "Electronics"

    ids, documents, metadatas = [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Processing {category}")):
            rec = json.loads(line)

            if not rec.get("title") or not rec.get("text"):
                continue

            record_id = f"{category}_{i}"
            document = f"{rec['title']}. {rec['text']}"
            if not isinstance(document, str) or not document.strip():
                continue

            metadata = {
                "rating": float(rec["rating"]) if rec.get("rating") is not None else 0.0,
                "title": rec["title"] if isinstance(rec.get("title"), str) else "Unknown",
                "main_category": rec["main_category"] if isinstance(rec.get("main_category"), str) else "Unknown"
            }

            ids.append(record_id)
            documents.append(document)
            metadatas.append(metadata)

    print(f"Aggiungo {len(ids)} record dal file {category}.jsonl nella collection 'amazon-reviews'...")

    BATCH_SIZE = 500
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch_docs).tolist()

        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            embeddings=batch_embeddings,
            metadatas=metadatas[i:i + BATCH_SIZE],
            documents=batch_docs
        )

    print("✅ Upload completato")

if __name__ == "__main__":
    main()
