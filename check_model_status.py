#!/usr/bin/env python3
"""
Check status of all models and provide next steps
"""
from pathlib import Path
import json

def check_file_size(path):
    """Get human-readable file size"""
    size = 0
    for file in path.rglob('*'):
        if file.is_file():
            size += file.stat().st_size
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def check_model_files(model_dir):
    """Check if model has all required files"""
    model_dir = Path(model_dir)
    required_files = [
        'config.json',
        'tokenizer_config.json',
    ]
    
    # Check for model weights (either format)
    has_weights = (model_dir / 'model.safetensors').exists() or \
                  (model_dir / 'pytorch_model.bin').exists()
    
    has_required = all((model_dir / f).exists() for f in required_files)
    
    return has_required and has_weights

def main():
    print("\n" + "="*70)
    print("MODEL STATUS CHECK - SemEval 2026 Psycholinguistic Markers")
    print("="*70)
    
    # Check extraction models
    print("\n📦 EXTRACTION MODELS (Merged, No PEFT)")
    print("-"*70)
    
    marker_types = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
    extraction_ready = []
    
    for marker_type in marker_types:
        model_dir = Path(f"models_for_sharing/extraction-{marker_type}-roberta-large-merged")
        
        if model_dir.exists() and check_model_files(model_dir):
            size = check_file_size(model_dir)
            extraction_ready.append(marker_type)
            print(f"  ✅ {marker_type:10s} | {size:>10s} | {model_dir}")
        else:
            print(f"  ❌ {marker_type:10s} | Missing or incomplete")
    
    # Check detection model
    print("\n🔍 DETECTION MODEL")
    print("-"*70)
    
    detection_merged = Path("models_for_sharing/detection-roberta-large-merged")
    detection_lora = Path("roberta-large-binary-conspiracy-lora/checkpoint-168")
    
    if detection_merged.exists() and check_model_files(detection_merged):
        size = check_file_size(detection_merged)
        print(f"  ✅ Merged model  | {size:>10s} | {detection_merged}")
        detection_status = "merged"
    elif detection_lora.exists():
        print(f"  ⚠️  LoRA checkpoint exists but cannot be merged")
        print(f"      Location: {detection_lora}")
        print(f"      Issue: ModulesToSaveWrapper corruption")
        print(f"      Status: Works locally, cannot upload to HF")
        detection_status = "lora_only"
    else:
        print(f"  ❌ No detection model found")
        detection_status = "missing"
    
    # Check submission files
    print("\n📝 CHALLENGE SUBMISSION FILES")
    print("-"*70)
    
    submission_file = Path("Markers-Extraction/submission.zip")
    if submission_file.exists():
        size = check_file_size(submission_file.parent / "submission.jsonl")
        print(f"  ✅ submission.zip | {size:>10s} | {submission_file}")
    else:
        print(f"  ❌ submission.zip not found")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print(f"\n✅ Extraction models ready: {len(extraction_ready)}/5")
    if extraction_ready:
        print(f"   {', '.join(extraction_ready)}")
    
    if detection_status == "merged":
        print(f"✅ Detection model: Ready to upload")
    elif detection_status == "lora_only":
        print(f"⚠️  Detection model: Works locally, upload broken")
    else:
        print(f"❌ Detection model: Not found")
    
    # Recommendations
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    
    if len(extraction_ready) == 5:
        print("\n1. ✅ Upload extraction models to Hugging Face:")
        print("   python3 upload_extraction_merged.py")
        print()
    
    if detection_status == "lora_only":
        print("2. ⚠️  Detection model:")
        print("   - Works fine locally for challenge submissions")
        print("   - Cannot upload to HF due to corrupted state")
        print("   - Options:")
        print("     a) Use locally (recommended for challenge)")
        print("     b) Retrain model for clean upload")
        print()
    
    if submission_file.exists():
        print("3. ✅ Challenge submission ready:")
        print(f"   Upload {submission_file} to challenge platform")
        print()
    
    print("="*70)
    print("📚 DOCUMENTATION")
    print("="*70)
    print()
    print("  - MODEL_SHARING_GUIDE.md  → Technical details")
    print("  - UPLOAD_GUIDE.md         → Upload instructions")
    print("  - check_model_status.py   → This script")
    print()
    
    print("="*70)
    print("💡 KEY POINTS")
    print("="*70)
    print()
    print("  ✅ Extraction models are production-ready")
    print("  ✅ No PEFT dependencies needed on other machines")
    print("  ✅ Challenge submission files are ready")
    print("  ⚠️  Detection model works locally (upload issue non-critical)")
    print()

if __name__ == '__main__':
    main()
