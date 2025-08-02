from typing import Optional
from dataclasses import dataclass
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
import chromadb
import sys
import os
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


@dataclass
class QueryReviewsInput:
    """Input parameters for querying similar reviews from ChromaDB."""
    query_text: str
    top_k: Optional[int] = 8


@tool
def query_reviews(input: QueryReviewsInput) -> str:
    """
    Queries the 'amazon-reviews' collection in ChromaDB and retrieves the most similar reviews.

    Args:
        input (QueryReviewsInput): An object containing the search query and the number of results to retrieve.

    Returns:
        str: A JSON string containing the list of the most similar reviews and their metadata.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="/app/chromadb/chroma_store")
    collection = client.get_or_create_collection("amazon-reviews")

    query_embedding = model.encode([input.query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=input.top_k,
        include=["metadatas", "documents", "distances"]
    )


    reviews = []
    for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        reviews.append({
            "document": doc,
            "metadata": metadata,
            "similarity_score": 1 - distance
        })

    return json.dumps(reviews, ensure_ascii=False, indent=2)
