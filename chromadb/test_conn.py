import os
import chromadb

chroma_path = os.path.abspath("chroma_store")
print(f"[INFO] Path completo usato per ChromaDB: {chroma_path}")


if os.path.isdir(chroma_path):
    print(f"[✓] Directory '{chroma_path}' trovata. Contenuto:")
    for f in os.listdir(chroma_path):
        print(f"  └─ {f}")
else:
    print(f"[❌] Directory '{chroma_path}' NON esiste.")

# Inizializza il client
client = chromadb.PersistentClient(path=chroma_path)

# Lista tutte le collection esistenti
collections = client.list_collections()
print(f"[INFO] Collections esistenti nel DB:")
for c in collections:
    print(f"  - {c.name}")

# Controlla se la collection 'amazon-reviews' esiste
try:
    collection = client.get_collection("amazon-reviews")
except Exception as e:
    print(f"[❌] Collection 'amazon-reviews' non trovata: {e}")
    collection = None

# Se esiste, stampa info
if collection:
    count = collection.count()
    print(f"[✓] Collection 'amazon-reviews' trovata. Documenti presenti: {count}")

    if count > 0:
        print("[INFO] Recupero i primi 5 documenti per ispezione...")
        results = collection.peek(5)
        for i, doc in enumerate(results["documents"]):
            print(f"\nDocumento {i + 1}:")
            print(f"  Text: {doc[:80]}...")
            print(f"  Metadata: {results['metadatas'][i]}")
else:
    print("[!] Collection non trovata, oppure vuota.")
