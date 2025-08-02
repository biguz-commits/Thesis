from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import torch


MODEL_PATH = Path("bert-nlp-classifier/final").resolve()

def classify_query(query: str) -> int:
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_PATH))

    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
    return predicted

if __name__ == "__main__":
    test_queries = [
        "What happen in 1945 in Japan?"
    ]

    for q in test_queries:
        label = classify_query(q)
        print(f"Query: {q}\n→ Predicted label: {label}\n")
