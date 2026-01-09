#!/usr/bin/env python3
"""
Inference script for marker extraction using trained RoBERTa-LoRA model
"""
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from peft import PeftModel

def load_data(file_path):
    """Load JSONL data"""
    data = []
    with open(file_path) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def load_model(model_dir):
    """Load trained model"""
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(model_dir)
    
    # Load label mapping
    with open(Path(model_dir) / 'label_mapping.json') as f:
        label_mapping = json.load(f)
    
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    model.eval()
    return model, tokenizer, id_to_label

def extract_markers_from_tokens(text, token_offsets, predicted_labels, id_to_label):
    """Extract marker spans from token predictions"""
    markers = []
    current_marker = None
    
    for token_idx, (offset, label_id) in enumerate(zip(token_offsets, predicted_labels)):
        if offset[0] is None or offset[1] is None:
            continue
        
        label = id_to_label[label_id]
        
        if label.startswith('B-'):
            # Save previous marker if exists
            if current_marker:
                markers.append(current_marker)
            
            # Start new marker
            marker_type = label[2:]  # Remove 'B-'
            current_marker = {
                'startIndex': offset[0],
                'endIndex': offset[1],
                'type': marker_type
            }
        
        elif label.startswith('I-'):
            # Continue current marker
            if current_marker and label[2:] == current_marker['type']:
                current_marker['endIndex'] = offset[1]
            # If no current marker or different type, skip
        
        else:  # 'O' label
            # Save previous marker if exists
            if current_marker:
                markers.append(current_marker)
                current_marker = None
    
    # Save last marker if exists
    if current_marker:
        markers.append(current_marker)
    
    return markers

def predict(model, tokenizer, id_to_label, examples, max_length=256):
    """Generate predictions for examples"""
    predictions = []
    
    for example in examples:
        text = example['text']
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0].tolist()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predicted labels
        predicted_labels = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        
        # Extract markers
        markers = extract_markers_from_tokens(
            text, offset_mapping, predicted_labels, id_to_label
        )
        
        predictions.append({
            '_id': example['_id'],
            'markers': markers
        })
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Marker extraction inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file')
    parser.add_argument('--model', type=str,
                       default='models/roberta-base-markers-lora/final_model',
                       help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (currently only 1 supported)')
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / args.model if not Path(args.model).is_absolute() else Path(args.model)
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    print('='*60)
    print('Marker Extraction Inference')
    print('='*60)
    print(f'Model: {model_dir}')
    print(f'Input: {input_file}')
    print(f'Output: {output_file}')
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print('='*60)
    
    # Load model
    print('Loading model...')
    model, tokenizer, id_to_label = load_model(model_dir)
    print('✓ Model loaded')
    
    # Load data
    print('Loading data...')
    data = load_data(input_file)
    print(f'✓ Loaded {len(data)} examples')
    
    # Predict
    print('Generating predictions...')
    predictions = predict(model, tokenizer, id_to_label, data)
    
    # Count markers
    total_markers = sum(len(p['markers']) for p in predictions)
    print(f'✓ Generated {total_markers} markers')
    
    # Save
    print('Saving predictions...')
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f'✓ Saved to: {output_file}')
    
    print('='*60)
    print('Inference complete!')
    print('='*60)

if __name__ == '__main__':
    main()
