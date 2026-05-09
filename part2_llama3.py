import json, re, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from groq import Groq

# CONFIG
SEED, DATASET, GROQ_MODEL = 42, "zeroshot/twitter-financial-news-sentiment", "llama-3.1-8b-instant"
LLM_RESULTS_JSON = "results_llm.json"

from data_utils import (
    LABEL2ID,
    load_and_split_data, 
    DEFAULT_N_TRAIN, 
    DEFAULT_N_VAL, 
    DEFAULT_N_TEST,
    normalize_label
)

def parse_robust_label(text: str) -> str:
    """Uses regex and keyword matching to find sentiment in messy LLM replies."""
    if not text: return "Neutral"
    for lab in ["Bearish", "Bullish", "Neutral"]:
        if re.search(rf"\b{lab}\b", text, flags=re.IGNORECASE): return lab
    t = text.lower()
    if any(k in t for k in ["bear", "negative"]): return "Bearish"
    if any(k in t for k in ["bull", "positive"]): return "Bullish"
    return "Neutral"

def fewshot_prompt(tweet: str) -> str:
    """Provides one example per class to satisfy project requirements."""
    examples = [
        ("Bearish", "Earnings missed estimates and guidance was cut. Stock likely goes down."),
        ("Bullish", "Revenue beat expectations and outlook improved. Stock likely goes up."),
        ("Neutral", "Company reported results in line with expectations. No clear direction."),
    ]
    
    p = [
        "You are a financial sentiment classifier. Return ONLY 'Bearish', 'Bullish', or 'Neutral'.",
        "\nExamples:"
    ]
    for lab, ex in examples:
        p.append(f"Tweet: {ex}\nLabel: {lab}")
    
    p.append(f"\nNow classify this tweet:\nTweet: {tweet}\nLabel:")
    return "\n".join(p)

def print_simple_table(ft, llm):
    """Prints a comparison table between the local model and the LLM."""
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"{'Metric':<15} | {'DistilBERT (Local)':<20} | {'Llama 3 (API)':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<15} | {ft['metrics']['accuracy']:<20.4f} | {llm['metrics']['accuracy']:<15.4f}")
    print(f"{'F1 (Macro)':<15} | {ft['metrics']['f1_macro']:<20.4f} | {llm['metrics']['f1_macro']:<15.4f}")
    print(f"{'Latency (ms)':<15} | {ft['latency_ms_per_sample']:<20.2f} | {llm['latency_ms_per_sample']:<15.2f}")
    print(f"{'Eval samples':<15} | {ft.get('eval_samples', 'N/A'):<20} | {llm['eval_samples']:<15}")

def main():
    api_key = input("Enter Groq API Key: ").strip() 
    with open("results_finetune.json", "r") as f: finetune = json.load(f)

    pack = load_and_split_data(
        dataset_name=DATASET,
        seed=SEED,
        n_train=DEFAULT_N_TRAIN,
        n_val=DEFAULT_N_VAL,
        n_test=DEFAULT_N_TEST,
    )

    test_raw = pack["test_raw"]
    gold = [normalize_label(ex[pack["label_field"]], pack["label_feature"]) for ex in test_raw]
    
    client = Groq(api_key=api_key)
    preds, latencies = [], []



    print(f"Evaluating {len(test_raw)} samples (Few-Shot)...")
    total_samples = len(test_raw)
    start_time = time.time()


    for i, ex in enumerate(test_raw, start=1):
        prompt = fewshot_prompt(ex[pack["text_field"]])
        
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0, 
            max_tokens=10
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        out = (resp.choices[0].message.content or "").strip()
        preds.append(LABEL2ID.get(parse_robust_label(out), 2))

        # --- DYNAMIC TERMINAL PRINTING ---
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        eta = avg_time * (total_samples - i)
        
        # \r moves the cursor back to the start of the line
        # end="" prevents a new line from being created
        print(f"\r Progress: {i}/{total_samples} [{i/total_samples:.1%}] | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s ", end="", flush=True)
        
    print("\n\n Evaluation Complete.")

    # Standardized JSON Schema
    llm_results = {
        "model": GROQ_MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "eval_samples": len(test_raw),
        "metrics": {
            "accuracy": float(accuracy_score(gold, preds)),
            "f1_macro": float(f1_score(gold, preds, average="macro"))
        },
        "latency_ms_per_sample": float(np.mean(latencies))
    }

    with open(LLM_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(llm_results, f, indent=4)
    
    print(f"\nSaved LLM results to {LLM_RESULTS_JSON}")
    print_simple_table(finetune, llm_results)

if __name__ == "__main__": main()