import json, time, torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, 
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
)

from data_utils import (
    ID2LABEL, LABEL2ID,
    load_and_split_data, 
    normalize_label,
    DEFAULT_N_TRAIN, 
    DEFAULT_N_VAL, 
    DEFAULT_N_TEST
)

# CONFIGURATION
SEED, DATASET, MODEL = 42, "zeroshot/twitter-financial-news-sentiment", "distilbert-base-uncased"

def prepare_for_trainer(raw_ds, tokenizer, text_f, label_f, label_feat):
    """Processes raw data into the format the Hugging Face Trainer expects."""

    # Step 1: Normalize labels and Step 2: Tokenize (dynamic padding)
    ds = raw_ds.map(lambda ex: {"labels": normalize_label(ex[label_f], label_feat)})
    ds = ds.map(lambda b: tokenizer(b[text_f], truncation=True), batched=True)
    
    # Step 3: Clean up columns
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"input_ids", "attention_mask", "labels"}])
    ds.set_format("torch")
    return ds

@torch.inference_mode()
def average_latency_ms(model, tokenizer, raw_test_ds, text_f):
    """Measures how fast our model processes one tweet (ms/sample)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    texts = [ex[text_f] for ex in raw_test_ds]
    loader = DataLoader(texts, batch_size=64)
    
    # Warmup pass 
    for batch in loader:
        enc = tokenizer(list(batch), padding=True, truncation=True, return_tensors="pt").to(device)
        _ = model(**enc)
        break

    start = time.perf_counter()
    n = 0
    for batch in loader:
        enc = tokenizer(list(batch), padding=True, truncation=True, return_tensors="pt").to(device)
        _ = model(**enc)
        n += len(batch)
    
    return (time.perf_counter() - start) * 1000.0 / max(n, 1)

def main():
    set_seed(SEED)
    print("--- Phase 1: Data Preparation ---")

    pack = load_and_split_data(
        dataset_name=DATASET,
        seed=SEED,
        n_train=DEFAULT_N_TRAIN,
        n_val=DEFAULT_N_VAL,
        n_test=DEFAULT_N_TEST,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train = prepare_for_trainer(pack["train_raw"], tokenizer, pack["text_field"], pack["label_field"], pack["label_feature"])
    val = prepare_for_trainer(pack["val_raw"], tokenizer, pack["text_field"], pack["label_field"], pack["label_feature"])
    test = prepare_for_trainer(pack["test_raw"], tokenizer, pack["text_field"], pack["label_field"], pack["label_feature"])

    # --- Training Phase ---
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            eval_strategy="epoch", save_strategy="no", #output_dir="ft_out", 
            num_train_epochs=3, per_device_train_batch_size=16, report_to=[]
        ),
        train_dataset=train, eval_dataset=val, tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, -1)),
                                   "f1_macro": f1_score(p.label_ids, np.argmax(p.predictions, -1), average="macro")}
    )
    trainer.train()

    # Save the model and tokenizer to a folder
    model_path = "./my_finetuned_sentiment_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Model saved to {model_path}")



    # --- Metrics & Saving ---
    m = trainer.evaluate(test)
    
    # Standardized JSON Schema
    results = {
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "eval_samples": len(pack["test_raw"]),
        "metrics": {
            "accuracy": float(m["eval_accuracy"]), 
            "f1_macro": float(m["eval_f1_macro"])
        },
        "latency_ms_per_sample": float(average_latency_ms(model, tokenizer, pack["test_raw"], pack["text_field"]))
    }

    with open("results_finetune.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved -> results_finetune.json")

if __name__ == "__main__": main()