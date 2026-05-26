import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CACHE_DIR, COMMON_MODELS

def download_model(model_name: str) -> bool:
    full_model_name = COMMON_MODELS.get(model_name, model_name)
    if model_name in COMMON_MODELS:
        print(f"Using model: {model_name} -> {full_model_name}")
    
    save_path = os.path.join(CACHE_DIR, "models/lm-bff/", model_name)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {full_model_name}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Save path: {save_path}")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=CACHE_DIR)
    model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=CACHE_DIR)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path, safe_serialization=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Download completed: {os.path.abspath(save_path)}")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained language models")
    parser.add_argument("--model", type=str, required=True, help="Model name to download")
    args = parser.parse_args()
    
    download_model(args.model)