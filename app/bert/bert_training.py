import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)



df = pd.read_csv("data.csv")


train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Modello
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. TrainingArguments
training_args = TrainingArguments(
    output_dir="bert-nlp-classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./bert-nlp-classifier/logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)


trainer.train()


trainer.save_model("./bert-nlp-classifier/final")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("./bert-nlp-classifier/final")
