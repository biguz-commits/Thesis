import os
import json
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    # Inizializza il modello locale
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ✅ Inizializza ChromaDB persistente (nuova API)
    client = chromadb.PersistentClient(path="chroma_store")
    collection = client.get_or_create_collection("amazon-reviews")

    processed_dir = os.path.join("..", "data", "processed", "All_Beauty.jsonl")
    files = [f for f in os.listdir(processed_dir) if f.endswith('.jsonl')]

    for file in files:
        category = file.replace('.jsonl', '')
        file_path = os.path.join(processed_dir, file)

        ids = []
        documents = []
        metadatas = []

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc=f"Processing {file}")):
                rec = json.loads(line)

                if not rec.get("title") or not rec.get("text"):
                    continue

                record_id = f"{category}_{i}"
                document = f"{rec['title']}. {rec['text']}"
                if not isinstance(document, str) or not document.strip():
                    continue

                metadata = {
                    "rating": float(rec["rating"]),
                    "title": rec["title"],
                    "main_category": rec["main_category"]
                }

                ids.append(record_id)
                documents.append(document)
                metadatas.append(metadata)

        print(f"Aggiungo {len(ids)} record dal file {file} nella collection 'amazon-reviews'...")

        # EMBEDDING LOCALE
        BATCH_SIZE = 500
        for i in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[i:i+BATCH_SIZE]
            batch_embeddings = model.encode(batch_docs).tolist()

            collection.add(
                ids=ids[i:i+BATCH_SIZE],
                embeddings=batch_embeddings,
                metadatas=metadatas[i:i+BATCH_SIZE],
                documents=batch_docs
            )

    print("✅ Dati caricati e salvati nella collection persistente 'amazon-reviews'")

if __name__ == "__main__":
    main()

