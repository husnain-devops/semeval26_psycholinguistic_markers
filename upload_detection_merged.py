#!/usr/bin/env python3
"""
Upload detection model to Hugging Face Hub
"""
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from huggingface_hub import login
from datetime import datetime
import os

def main():
    # Get username
    hf_user = os.environ.get('HF_USERNAME', 'MMHusnain')
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    repo_id = f"{hf_user}/semeval26-detection-{timestamp}"
    
    print("="*70)
    print("UPLOADING DETECTION MODEL")
    print("="*70)
    print(f"User: {hf_user}")
    print(f"Repo: {repo_id}\n")
    
    # Login
    print("Logging into Hugging Face...")
    login()
    
    # Load model
    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained("models_for_sharing/detection-roberta-large-merged")
    tokenizer = RobertaTokenizerFast.from_pretrained("models_for_sharing/detection-roberta-large-merged")
    
    # Upload
    print("Pushing model to Hub...")
    model.push_to_hub(repo_id, private=False)
    
    print("Pushing tokenizer to Hub...")
    tokenizer.push_to_hub(repo_id)
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"\n🌐 Model URL: https://huggingface.co/{repo_id}")
    print(f"\n💡 To use on another machine:")
    print(f"   model = RobertaForSequenceClassification.from_pretrained('{repo_id}')")

if __name__ == '__main__':
    main()
