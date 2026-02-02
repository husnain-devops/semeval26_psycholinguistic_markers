#!/usr/bin/env python3
"""
Fix adapter_config.json files to be compatible with older PEFT versions
Removes problematic fields like 'alora_invocation_tokens'
"""
import json
import glob
from pathlib import Path

def clean_adapter_config(config_path):
    """Keep only essential LoRA fields, remove all others"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Keep only these essential fields
    essential_fields = {
        'peft_type',
        'base_model_name_or_path',
        'r',
        'lora_alpha',
        'lora_dropout',
        'target_modules',
        'bias',
        'fan_in_fan_out',
        'init_lora_weights',
        'layers_to_transform',
        'layers_pattern',
        'modules_to_save',
        'inference_mode',
        'task_type'
    }
    
    # Create new config with only essential fields
    new_config = {}
    removed_fields = []
    
    for key, value in config.items():
        if key in essential_fields:
            new_config[key] = value
        else:
            removed_fields.append(key)
    
    if removed_fields:
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        print(f"  Removed {len(removed_fields)} incompatible fields")
        return True
    return False

def main():
    base_dir = Path('.')
    
    print('='*70)
    print('FIXING ADAPTER CONFIG FILES FOR PEFT COMPATIBILITY')
    print('='*70)
    
    # Find all adapter_config.json files
    adapter_configs = []
    
    # Detection model
    adapter_configs.extend(glob.glob('roberta-large-binary-conspiracy-lora/**/adapter_config.json', recursive=True))
    
    # Extraction models
    adapter_configs.extend(glob.glob('Markers-Extraction/models/**/adapter_config.json', recursive=True))
    
    print(f'\nFound {len(adapter_configs)} adapter config files\n')
    
    for config_path in adapter_configs:
        print(f'Processing: {config_path}')
        if clean_adapter_config(config_path):
            print(f'  ✓ Fixed')
        else:
            print(f'  ✓ Already compatible')
    
    print('\n' + '='*70)
    print('✅ ALL ADAPTER CONFIGS FIXED!')
    print('='*70)
    print('\nNow you can run:')
    print('  python3 upload_detection_hf.py')
    print('  python3 upload_extraction_5models_hf.py')

if __name__ == '__main__':
    main()
