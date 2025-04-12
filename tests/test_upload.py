import chromadb
from sentence_transformers import SentenceTransformer

def main():
    # Inizializza il modello di embedding locale
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ✅ Inizializza ChromaDB in modalità persistente (nuova API)
    client = chromadb.PersistentClient(path="../db/chroma_store")
    collection = client.get_or_create_collection("amazon-reviews")

    # 📊 Conta i documenti presenti nella collection
    total_docs = collection.count()
    print(f"\n📊 Totale documenti nella collection: {total_docs}")

    # 🔍 Stampa un sample di documenti (max 3)
    if total_docs > 0:
        sample = collection.get(limit=3, include=["documents", "metadatas"])
        for i in range(len(sample["documents"])):
            print(f"\n📦 Documento {i+1}")
            print("📝 Testo:", sample["documents"][i])
            print("🧾 Metadata:", sample["metadatas"][i])
    else:
        print("⚠️ Nessun documento trovato nella collection.")
        return

    # 🔎 Query test
    query = "This spray is really nice. It smells really good."  # Puoi cambiarla
    print(f"\n🔍 Eseguo la query: '{query}'")

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print("\n✅ Risultati della query:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n🔹 Risultato {i+1}")
        print("📄 Documento:", doc)
        print("📌 Metadata:", meta)
        print("📏 Distanza:", round(dist, 4))

if __name__ == "__main__":
    main()
