import json
import chromadb
from tools.vectordb_tool import query_reviews, QueryReviewsInput

def test_or_upload_collection():
    client = chromadb.PersistentClient(path="/chromadb/chroma_store")
    collection = client.get_or_create_collection("amazon-reviews")

    count = collection.count()
    if count == 0:
        print("[⚠️] La collection è vuota. Eseguo l'upload dei dati...")


def test_query_reviews():
    # Input in formato dizionario
    input_data = {
        "input": {
            "query_text": "fantasy books",
            "top_k": 3
        }
    }

    # Esecuzione del tool come funzione LangChain Tool (decorato con @tool)
    result_json = query_reviews(input_data)
    results = json.loads(result_json)

    # Controlli
    assert isinstance(results, list) and len(results) > 0, "Nessuna recensione trovata."
    for review in results:
        assert "document" in review and "similarity_score" in review, "Formato della risposta non valido."

    print(f"[✓] Query riuscita: trovate {len(results)} recensioni simili.")
    for i, r in enumerate(results, 1):
        print(f"  {i}. Score: {r['similarity_score']:.3f} | Title: {r['metadata'].get('title', 'N/A')[:60]}")

if __name__ == "__main__":
    test_or_upload_collection()
    test_query_reviews()
