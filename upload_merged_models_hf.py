#!/usr/bin/env python3
"""
Upload merged models (no PEFT) to Hugging Face Hub
These models are portable and work on any machine
"""
from pathlib import Path
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from huggingface_hub import HfApi, login
from datetime import datetime
import os

def upload_extraction_models():
    """Upload 5 extraction models"""
    marker_types = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
    
    # Get username
    hf_user = os.environ.get('HF_USERNAME', 'MMHusnain')
    
    print("="*70)
    print("UPLOADING EXTRACTION MODELS (MERGED - NO PEFT)")
    print("="*70)
    print(f"User: {hf_user}\n")
    
    # Login
    login()
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    
    for marker_type in marker_types:
        model_dir = Path(f"models_for_sharing/extraction-{marker_type}-roberta-large-merged")
        
        if not model_dir.exists():
            print(f"⚠ {marker_type}: Model not found at {model_dir}")
            continue
        
        repo_id = f"{hf_user}/semeval26-extraction-{marker_type}-{timestamp}"
        
        print(f"\n[{marker_type}]")
        print(f"  Loading from: {model_dir}")
        print(f"  Uploading to: {repo_id}")
        
        # Load model
        model = RobertaForTokenClassification.from_pretrained(str(model_dir))
        tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir))
        
        # Upload
        print(f"  Pushing model...")
        model.push_to_hub(repo_id, private=False)
        
        print(f"  Pushing tokenizer...")
        tokenizer.push_to_hub(repo_id)
        
        print(f"  ✅ Uploaded!")
        print(f"  🌐 https://huggingface.co/{repo_id}")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS UPLOADED!")
    print("="*70)
    print("\n💡 To use on another machine:")
    print("   from transformers import RobertaForTokenClassification")
    print(f"   model = RobertaForTokenClassification.from_pretrained('{hf_user}/semeval26-extraction-Action-{timestamp}')")

if __name__ == '__main__':
    upload_extraction_models()
