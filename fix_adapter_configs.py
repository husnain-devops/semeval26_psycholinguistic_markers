#!/usr/bin/env python3
"""
Fix adapter_config.json files to be compatible with older PEFT versions
Removes problematic fields like 'alora_invocation_tokens'
"""
import json
import glob
from pathlib import Path

def clean_adapter_config(config_path):
    """Remove incompatible fields from adapter_config.json"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fields to remove (incompatible with older PEFT versions)
    fields_to_remove = [
        'alora_invocation_tokens',
        'corda_config',
        'eva_config',
        'arrow_config',
        'qalora_group_size'
    ]
    
    modified = False
    for field in fields_to_remove:
        if field in config:
            del config[field]
            modified = True
            print(f"  Removed: {field}")
    
    if modified:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
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
