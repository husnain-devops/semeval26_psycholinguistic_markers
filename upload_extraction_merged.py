#!/usr/bin/env python3
"""
Upload merged extraction models (no PEFT dependencies) to Hugging Face Hub
These models are portable and work on any machine
"""
from pathlib import Path
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from huggingface_hub import login
from datetime import datetime
import os

def upload_extraction_models():
    """Upload 5 merged extraction models"""
    marker_types = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
    
    # Get username
    hf_user = os.environ.get('HF_USERNAME', 'MMHusnain')
    
    print("="*70)
    print("UPLOADING EXTRACTION MODELS (MERGED - NO PEFT)")
    print("="*70)
    print(f"User: {hf_user}\n")
    
    # Login
    print("Logging into Hugging Face...")
    try:
        login()
        print("✓ Logged in successfully\n")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("Please run: huggingface-cli login")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    uploaded_models = []
    
    for marker_type in marker_types:
        model_dir = Path(f"models_for_sharing/extraction-{marker_type}-roberta-large-merged")
        
        if not model_dir.exists():
            print(f"⚠ {marker_type}: Model not found at {model_dir}")
            continue
        
        repo_id = f"{hf_user}/semeval26-extraction-{marker_type}-{timestamp}"
        
        print(f"\n{'='*70}")
        print(f"[{marker_type}]")
        print(f"{'='*70}")
        print(f"  Loading from: {model_dir}")
        print(f"  Uploading to: {repo_id}")
        
        try:
            # Load model
            print(f"  Loading model...")
            model = RobertaForTokenClassification.from_pretrained(str(model_dir))
            
            # Load tokenizer
            print(f"  Loading tokenizer...")
            tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir))
            
            # Upload model
            print(f"  Pushing model to Hub...")
            model.push_to_hub(repo_id, private=False)
            
            # Upload tokenizer
            print(f"  Pushing tokenizer to Hub...")
            tokenizer.push_to_hub(repo_id)
            
            print(f"  ✅ Uploaded successfully!")
            print(f"  🌐 https://huggingface.co/{repo_id}")
            
            uploaded_models.append({
                'marker_type': marker_type,
                'repo_id': repo_id,
                'local_path': str(model_dir)
            })
            
        except Exception as e:
            print(f"  ❌ Error uploading {marker_type}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    
    if uploaded_models:
        print(f"\n✅ Successfully uploaded {len(uploaded_models)} models:\n")
        for m in uploaded_models:
            print(f"  [{m['marker_type']}] {m['repo_id']}")
        
        print("\n" + "="*70)
        print("💡 TO USE ON ANOTHER MACHINE:")
        print("="*70)
        print("\n1. Install dependencies:")
        print("   pip install transformers torch\n")
        
        print("2. Load model:")
        print("   from transformers import RobertaForTokenClassification")
        
        if uploaded_models:
            example = uploaded_models[0]
            print(f"   model = RobertaForTokenClassification.from_pretrained('{example['repo_id']}')\n")
        
        print("3. Run inference:")
        print("   from transformers import pipeline")
        print(f"   ner = pipeline('ner', model=model, tokenizer=tokenizer)")
        print("   results = ner('Your text here')\n")
    else:
        print("\n❌ No models were uploaded")
        print("   Please check the errors above")

if __name__ == '__main__':
    upload_extraction_models()
