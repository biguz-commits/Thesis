from transformers import BertForSequenceClassification, AutoTokenizer
import torch
from llm.state_types import State  
from pathlib import Path


model_path = Path("./bert/bert-nlp-classifier/final").resolve()

if not model_path.exists():
    raise FileNotFoundError(f"❌ Model path not found: {model_path}")

model = BertForSequenceClassification.from_pretrained(
    str(model_path),
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    local_files_only=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def create_bert_classifier_node():
    def classify_input(state: State) -> State:
        input_text = state["input_query"]
        if not input_text:
            raise ValueError("Missing 'input_query' in state")

        encoded = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        state["label"] = prediction
        return state

    return classify_input
