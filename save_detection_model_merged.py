#!/usr/bin/env python3
"""
Save detection model in merged format (no PEFT dependencies)
This script handles the conspiracy detection model specifically
"""
import json
import torch
from pathlib import Path
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast
)
from peft import PeftModel

def find_best_checkpoint(base_dir):
    """Find the best checkpoint (highest step number)"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    checkpoints = list(base_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]), reverse=True)
    return checkpoints[0]

def merge_and_save_detection_model():
    """Merge LoRA weights into base model for detection task"""
    print("="*70)
    print("SAVING DETECTION MODEL (MERGED)")
    print("="*70)
    
    # Try to find the checkpoint
    checkpoint_dir = find_best_checkpoint("roberta-large-binary-conspiracy-lora")
    
    if checkpoint_dir is None:
        # Try alternative location
        checkpoint_dir = find_best_checkpoint("EDA-Rehydrated/notebooks/roberta-large-binary-conspiracy-lora")
    
    if checkpoint_dir is None:
        print("❌ No checkpoint found!")
        print("   Tried:")
        print("     - roberta-large-binary-conspiracy-lora/checkpoint-*")
        print("     - EDA-Rehydrated/notebooks/roberta-large-binary-conspiracy-lora/checkpoint-*")
        return None
    
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading from: {checkpoint_dir}")
    
    try:
        # Load base model
        print("  Loading base RoBERTa-large model...")
        base_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-large",
            num_labels=3
        )
        
        # Load PEFT model
        print("  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        
        # Merge LoRA weights into base model
        print("  Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        print(f"  Saving merged model to: {output_dir}")
        merged_model.save_pretrained(str(output_dir))
        
        # Save tokenizer
        print("  Saving tokenizer...")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        tokenizer.save_pretrained(str(output_dir))
        
        # Save label mapping
        label_mapping = {
            'label_to_id': {'cant_tell': 0, 'no': 1, 'yes': 2},
            'id_to_label': {0: 'cant_tell', 1: 'no', 2: 'yes'},
            'num_labels': 3
        }
        with open(output_dir / 'label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"✅ Detection model saved (merged, no LoRA)")
        print(f"   Location: {output_dir}")
        print(f"   Can be loaded on any machine with:")
        print(f"   model = RobertaForSequenceClassification.from_pretrained('{output_dir}')")
        
        return str(output_dir)
        
    except Exception as e:
        print(f"❌ Error merging detection model: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: Load from trainer state if available
        try:
            trainer_state_file = checkpoint_dir / "trainer_state.json"
            if trainer_state_file.exists():
                print("  Found trainer state, attempting direct load...")
                
                # Try loading the model directly without PEFT
                model = RobertaForSequenceClassification.from_pretrained(
                    str(checkpoint_dir),
                    num_labels=3
                )
                
                print(f"  Saving model to: {output_dir}")
                model.save_pretrained(str(output_dir))
                
                tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
                tokenizer.save_pretrained(str(output_dir))
                
                label_mapping = {
                    'label_to_id': {'cant_tell': 0, 'no': 1, 'yes': 2},
                    'id_to_label': {0: 'cant_tell', 1: 'no', 2: 'yes'},
                    'num_labels': 3
                }
                with open(output_dir / 'label_mapping.json', 'w') as f:
                    json.dump(label_mapping, f, indent=2)
                
                print(f"✅ Detection model saved (direct load)")
                return str(output_dir)
                
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            return None

def main():
    print("\n" + "="*70)
    print("DETECTION MODEL SAVING FOR CROSS-MACHINE COMPATIBILITY")
    print("="*70)
    print("\nThis script merges LoRA weights into base model")
    print("Result: Standard PyTorch model (no PEFT dependency)")
    print("="*70)
    
    detection_path = merge_and_save_detection_model()
    
    # Summary
    print("\n" + "="*70)
    print("📦 SUMMARY")
    print("="*70)
    
    if detection_path:
        print(f"\n✅ Detection model saved to:")
        print(f"   {detection_path}")
        print("\n✅ This model can be used on ANY machine:")
        print("   - No PEFT required")
        print("   - Just need: pip install transformers torch")
        print("   - Load with:")
        print(f"     model = RobertaForSequenceClassification.from_pretrained('{detection_path}')")
        print("\n💡 To upload to Hugging Face:")
        print("   from transformers import RobertaForSequenceClassification")
        print(f"   model = RobertaForSequenceClassification.from_pretrained('{detection_path}')")
        print("   model.push_to_hub('your-username/your-repo-name')")
    else:
        print("\n❌ Detection model could not be saved")
        print("   Please check the errors above")
        print("\n💡 Alternative: You can retrain the model or use a different checkpoint")

if __name__ == '__main__':
    main()
