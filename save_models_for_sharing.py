#!/usr/bin/env python3
"""
Save models in a format that's portable across machines/PEFT versions
Option 1: Merge LoRA weights into base model (recommended for sharing)
Option 2: Save minimal adapter configs
"""
import json
import torch
from pathlib import Path
from transformers import (
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaTokenizerFast
)
from peft import PeftModel

def merge_and_save_detection_model():
    """Merge LoRA weights into base model for detection task"""
    print("="*70)
    print("SAVING DETECTION MODEL (MERGED)")
    print("="*70)
    
    checkpoint_dir = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading from: {checkpoint_dir}")
    
    # Load base model
    base_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=3
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    
    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(str(output_dir))
    
    # Save tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"✅ Detection model saved (merged, no LoRA)")
    print(f"   Can be loaded on any machine with: ")
    print(f"   model = RobertaForSequenceClassification.from_pretrained('{output_dir}')")
    
    return str(output_dir)

def merge_and_save_extraction_models():
    """Merge LoRA weights for all 5 extraction models"""
    print("\n" + "="*70)
    print("SAVING EXTRACTION MODELS (MERGED)")
    print("="*70)
    
    marker_types = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
    saved_paths = {}
    
    for marker_type in marker_types:
        print(f"\nProcessing {marker_type}...")
        
        model_dir = Path(f"Markers-Extraction/models/roberta-large-lora-{marker_type}/final_model")
        output_dir = Path(f"models_for_sharing/extraction-{marker_type}-roberta-large-merged")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load base model
            base_model = RobertaForTokenClassification.from_pretrained(
                "roberta-large",
                num_labels=2
            )
            
            # Load PEFT model
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            
            # Merge LoRA weights
            print(f"  Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            
            # Save
            print(f"  Saving to: {output_dir}")
            merged_model.save_pretrained(str(output_dir))
            
            # Save tokenizer
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", add_prefix_space=True)
            tokenizer.save_pretrained(str(output_dir))
            
            # Save label mapping
            label_mapping = {
                'label_to_id': {'O': 0, marker_type: 1},
                'id_to_label': {0: 'O', 1: marker_type}
            }
            with open(output_dir / 'label_mapping.json', 'w') as f:
                json.dump(label_mapping, f, indent=2)
            
            print(f"  ✅ {marker_type} saved")
            saved_paths[marker_type] = str(output_dir)
            
        except Exception as e:
            print(f"  ❌ Error saving {marker_type}: {e}")
    
    return saved_paths

def main():
    print("\n" + "="*70)
    print("MODEL SAVING FOR CROSS-MACHINE COMPATIBILITY")
    print("="*70)
    print("\nThis script merges LoRA weights into base models")
    print("Result: Standard PyTorch models (no PEFT dependency)")
    print("Can be loaded on ANY machine with just transformers library")
    print("="*70)
    
    # Check if models exist
    detection_checkpoint = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    if not detection_checkpoint.exists():
        print("\n⚠ Detection model checkpoint not found")
        print(f"   Looking for: {detection_checkpoint}")
    else:
        try:
            detection_path = merge_and_save_detection_model()
        except Exception as e:
            print(f"❌ Error with detection model: {e}")
            detection_path = None
    
    # Check extraction models
    extraction_base = Path("Markers-Extraction/models")
    if extraction_base.exists():
        try:
            extraction_paths = merge_and_save_extraction_models()
        except Exception as e:
            print(f"❌ Error with extraction models: {e}")
            extraction_paths = {}
    
    # Summary
    print("\n" + "="*70)
    print("📦 SUMMARY - MODELS READY FOR SHARING")
    print("="*70)
    
    print("\nSaved models (merged, no LoRA dependencies):")
    print("  models_for_sharing/")
    
    if detection_path:
        print(f"    ├── detection-roberta-large-merged/")
    
    for marker_type in ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']:
        if extraction_paths.get(marker_type):
            print(f"    ├── extraction-{marker_type}-roberta-large-merged/")
    
    print("\n✅ These models can be used on ANY machine:")
    print("   - No PEFT required")
    print("   - Just need: pip install transformers torch")
    print("   - Load with: model = RobertaForSequenceClassification.from_pretrained(path)")
    
    print("\n💡 To upload to Hugging Face:")
    print("   Use the standard transformers upload:")
    print("   model.push_to_hub('your-repo-name')")

if __name__ == '__main__':
    main()
