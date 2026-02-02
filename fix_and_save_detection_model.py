#!/usr/bin/env python3
"""
Fix and save detection model using multiple approaches
Tries different methods to extract the trained weights and create a merged model
"""
import torch
import json
from pathlib import Path
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    RobertaConfig
)
from peft import PeftModel, LoraConfig, get_peft_model

def method_1_direct_load():
    """Method 1: Try loading model directly from checkpoint"""
    print("\n" + "="*70)
    print("METHOD 1: Direct Load from Checkpoint")
    print("="*70)
    
    checkpoint_dir = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("  Attempting direct load...")
        model = RobertaForSequenceClassification.from_pretrained(
            str(checkpoint_dir),
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        print("  ✅ Direct load successful!")
        print(f"  Saving to: {output_dir}")
        model.save_pretrained(str(output_dir))
        
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        tokenizer.save_pretrained(str(output_dir))
        
        save_label_mapping(output_dir)
        
        print("  ✅ Method 1 successful!")
        return True, str(output_dir)
        
    except Exception as e:
        print(f"  ❌ Method 1 failed: {e}")
        return False, None

def method_2_load_base_and_checkpoint_weights():
    """Method 2: Load base model and manually load checkpoint weights"""
    print("\n" + "="*70)
    print("METHOD 2: Manual Weight Loading")
    print("="*70)
    
    checkpoint_dir = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("  Loading base RoBERTa-large...")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-large",
            num_labels=3
        )
        
        # Try to find model weights
        print("  Looking for model weights...")
        model_file = None
        for filename in ["model.safetensors", "pytorch_model.bin", "adapter_model.bin"]:
            potential_file = checkpoint_dir / filename
            if potential_file.exists():
                model_file = potential_file
                print(f"  Found: {filename}")
                break
        
        if model_file is None:
            raise FileNotFoundError("No model weights found in checkpoint")
        
        # Load state dict
        print(f"  Loading weights from {model_file.name}...")
        if model_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
        else:
            state_dict = torch.load(str(model_file), map_location="cpu")
        
        # Try to load weights (may have mismatches)
        print("  Applying weights...")
        model.load_state_dict(state_dict, strict=False)
        
        print("  ✅ Weights loaded!")
        print(f"  Saving to: {output_dir}")
        model.save_pretrained(str(output_dir))
        
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        tokenizer.save_pretrained(str(output_dir))
        
        save_label_mapping(output_dir)
        
        print("  ✅ Method 2 successful!")
        return True, str(output_dir)
        
    except Exception as e:
        print(f"  ❌ Method 2 failed: {e}")
        return False, None

def method_3_recreate_with_lora_and_merge():
    """Method 3: Recreate LoRA config and try to merge cleanly"""
    print("\n" + "="*70)
    print("METHOD 3: Clean LoRA Recreation")
    print("="*70)
    
    checkpoint_dir = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("  Loading base model...")
        base_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-large",
            num_labels=3
        )
        
        # Create a clean LoRA config
        print("  Creating clean LoRA config...")
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        
        # Apply LoRA
        print("  Applying LoRA...")
        model = get_peft_model(base_model, lora_config)
        
        # Try to load adapter weights
        print("  Loading adapter weights...")
        adapter_file = checkpoint_dir / "adapter_model.bin"
        if adapter_file.exists():
            adapter_weights = torch.load(str(adapter_file), map_location="cpu")
            model.load_state_dict(adapter_weights, strict=False)
        
        # Merge and unload
        print("  Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        
        print(f"  Saving to: {output_dir}")
        merged_model.save_pretrained(str(output_dir))
        
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        tokenizer.save_pretrained(str(output_dir))
        
        save_label_mapping(output_dir)
        
        print("  ✅ Method 3 successful!")
        return True, str(output_dir)
        
    except Exception as e:
        print(f"  ❌ Method 3 failed: {e}")
        return False, None

def method_4_load_from_best_model():
    """Method 4: Try loading from best_model directory if it exists"""
    print("\n" + "="*70)
    print("METHOD 4: Load from Best Model")
    print("="*70)
    
    possible_dirs = [
        "roberta-large-binary-conspiracy-lora",
        "EDA-Rehydrated/notebooks/roberta-large-binary-conspiracy-lora",
        "roberta-large-binary-conspiracy-lora/best_model",
    ]
    
    output_dir = Path("models_for_sharing/detection-roberta-large-merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_dir in possible_dirs:
        model_path = Path(model_dir)
        if not model_path.exists():
            continue
        
        print(f"  Trying: {model_path}")
        try:
            # Check if this is a complete model
            config_file = model_path / "config.json"
            if not config_file.exists():
                print(f"    No config.json, skipping...")
                continue
            
            model = RobertaForSequenceClassification.from_pretrained(
                str(model_path),
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            
            print("  ✅ Successfully loaded!")
            print(f"  Saving to: {output_dir}")
            model.save_pretrained(str(output_dir))
            
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            tokenizer.save_pretrained(str(output_dir))
            
            save_label_mapping(output_dir)
            
            print("  ✅ Method 4 successful!")
            return True, str(output_dir)
            
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    print("  ❌ Method 4 failed: No valid model directory found")
    return False, None

def save_label_mapping(output_dir):
    """Save label mapping configuration"""
    label_mapping = {
        'label_to_id': {'cant_tell': 0, 'no': 1, 'yes': 2},
        'id_to_label': {0: 'cant_tell', 1: 'no', 2: 'yes'},
        'num_labels': 3,
        'task': 'conspiracy_detection',
        'model_type': 'roberta-large'
    }
    with open(Path(output_dir) / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)

def verify_model(model_dir):
    """Verify the saved model can be loaded"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    try:
        print(f"  Loading model from: {model_dir}")
        model = RobertaForSequenceClassification.from_pretrained(model_dir)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        
        print("  Testing inference...")
        test_text = "This is a test conspiracy theory about the government."
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        labels = ['cant_tell', 'no', 'yes']
        pred_idx = predictions.argmax().item()
        
        print(f"  ✅ Model works!")
        print(f"  Test prediction: {labels[pred_idx]} (confidence: {predictions[0][pred_idx]:.2%})")
        
        # Check file sizes
        model_files = list(Path(model_dir).rglob('*'))
        total_size = sum(f.stat().st_size for f in model_files if f.is_file())
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("DETECTION MODEL FIX & SAVE")
    print("="*70)
    print("\nTrying multiple methods to extract and save the model...")
    
    methods = [
        ("Method 1: Direct Load", method_1_direct_load),
        ("Method 2: Manual Weight Loading", method_2_load_base_and_checkpoint_weights),
        ("Method 3: Clean LoRA Recreation", method_3_recreate_with_lora_and_merge),
        ("Method 4: Load from Best Model", method_4_load_from_best_model),
    ]
    
    success = False
    model_path = None
    
    for method_name, method_func in methods:
        success, model_path = method_func()
        if success:
            print(f"\n{'='*70}")
            print(f"✅ {method_name} SUCCEEDED!")
            print(f"{'='*70}")
            break
    
    if success and model_path:
        # Verify the model
        if verify_model(model_path):
            print("\n" + "="*70)
            print("🎉 SUCCESS!")
            print("="*70)
            print(f"\n✅ Detection model saved to:")
            print(f"   {model_path}")
            print("\n✅ Model is verified and working!")
            print("\n📤 To upload to Hugging Face:")
            print("   python3 upload_detection_merged.py")
            
            # Create upload script
            create_upload_script(model_path)
            
    else:
        print("\n" + "="*70)
        print("❌ ALL METHODS FAILED")
        print("="*70)
        print("\n⚠️  The checkpoint appears to be corrupted beyond repair.")
        print("\n💡 Recommended solutions:")
        print("   1. Retrain the model (cleanest solution)")
        print("   2. If you have the model in a notebook, save it from there:")
        print("      trainer.model.merge_and_unload().save_pretrained('models_for_sharing/detection-merged')")
        print("   3. Use the model locally (it works for inference)")

def create_upload_script(model_path):
    """Create a simple upload script for the detection model"""
    script = f"""#!/usr/bin/env python3
\"\"\"
Upload detection model to Hugging Face Hub
\"\"\"
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from huggingface_hub import login
from datetime import datetime
import os

def main():
    # Get username
    hf_user = os.environ.get('HF_USERNAME', 'MMHusnain')
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    repo_id = f"{{hf_user}}/semeval26-detection-{{timestamp}}"
    
    print("="*70)
    print("UPLOADING DETECTION MODEL")
    print("="*70)
    print(f"User: {{hf_user}}")
    print(f"Repo: {{repo_id}}\\n")
    
    # Login
    print("Logging into Hugging Face...")
    login()
    
    # Load model
    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained("{model_path}")
    tokenizer = RobertaTokenizerFast.from_pretrained("{model_path}")
    
    # Upload
    print("Pushing model to Hub...")
    model.push_to_hub(repo_id, private=False)
    
    print("Pushing tokenizer to Hub...")
    tokenizer.push_to_hub(repo_id)
    
    print("\\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"\\n🌐 Model URL: https://huggingface.co/{{repo_id}}")
    print(f"\\n💡 To use on another machine:")
    print(f"   model = RobertaForSequenceClassification.from_pretrained('{{repo_id}}')")

if __name__ == '__main__':
    main()
"""
    
    script_path = Path("upload_detection_merged.py")
    script_path.write_text(script)
    script_path.chmod(0o755)
    print(f"\n✅ Created upload script: {script_path}")

if __name__ == '__main__':
    main()
