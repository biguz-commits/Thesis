import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
import pprint

# Carica variabili d'ambiente
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD')
PINECONE_REGION = os.getenv('PINECONE_REGION')
PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE')

# Inizializza Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Crea indice se non esiste, con embeddings integrati (llama-text-embed-v2)
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )

# Connessione all'indice
index = pc.Index(PINECONE_INDEX_NAME)


def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def prepare_records(records, category):
    prepared = []
    for i, rec in enumerate(records):
        # Utilizza il testo completo per generare l'embedding
        chunk_text = f"{rec['title']}. {rec['text']}"
        # Nei metadata mantieni solo i campi leggeri e utili per il filtering
        metadata = {
            "title": rec["title"],
            "rating": float(rec["rating"]),
            "main_category": rec["main_category"]
        }
        record = {
            "id": f"{category}_{i}",
            "values": {},  # Lasciamo vuoto perché Pinecone genera gli embeddings
            "chunk_text": chunk_text,
            "metadata": metadata
        }
        prepared.append(record)
    return prepared



def upsert_records(records, namespace, batch_size=90):
    for i in tqdm(range(0, len(records), batch_size), desc=f"Uploading batches to {namespace}"):
        batch = records[i:i + batch_size]
        index.upsert_records(namespace, batch)


if __name__ == "__main__":
    processed_dir = "../data/processed"
    files = [f for f in os.listdir(processed_dir) if f.endswith('.jsonl')]

    for file in files:
        category = file.replace('.jsonl', '')
        raw_records = load_jsonl(os.path.join(processed_dir, file))
        pinecone_records = prepare_records(raw_records, category)

        upsert_records(pinecone_records, PINECONE_NAMESPACE)

    print("✅ Upload completato con successo su Pinecone!")
