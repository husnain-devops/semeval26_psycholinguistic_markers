#!/usr/bin/env python3
"""
Run inference with 5 separate RoBERTa-large-LoRA models and combine results
"""
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import torch
import numpy as np
import zipfile
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset

# Configuration
BASE = Path(__file__).parent.parent.parent
TEST_FILE = BASE / 'dev_rehydrated.jsonl'
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
MAX_LENGTH = 128  # Match training

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


def tokenize_for_inference(examples, tokenizer):
    """Tokenize without labels for inference"""
    tok = tokenizer(
        examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH,
        return_offsets_mapping=True, is_split_into_words=False
    )
    # Dummy labels (required by dataset but not used)
    tok['labels'] = [[0] * len(offset) for offset in tok['offset_mapping']]
    return tok

def main():
    print('='*70)
    print('INFERENCE WITH 5 SEPARATE MODELS')
    print('='*70)
    print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    
    # Load test data
    print(f'\nLoading test data from: {TEST_FILE}')
    test_data = load_data(TEST_FILE)
    test_data_original = test_data.copy()
    print(f'✓ Loaded {len(test_data)} test examples')
    
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    
    # Run inference with each model
    all_predictions = {marker_type: [] for marker_type in MARKER_TYPES}
    
    for marker_idx, marker_type in enumerate(MARKER_TYPES):
        print(f'\n[{marker_idx+1}/5] Processing {marker_type}...')
        
        # Load model
        model_dir = BASE / 'Markers-Extraction' / 'models' / f'roberta-large-lora-{marker_type}' / 'final_model'
        print(f'  Loading from: {model_dir}')
        
        model = RobertaForTokenClassification.from_pretrained(model_dir)
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        # Load label mapping
        with open(model_dir / 'label_mapping.json') as f:
            label_mapping = json.load(f)
            id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
        
        # Tokenize test data
        test_ds = Dataset.from_list(test_data)
        cols_to_remove = [col for col in ['_id', 'text', 'markers', 'subreddit', 'conspiracy', 'annotator'] 
                          if col in test_ds.column_names]
        
        test_ds_tok = test_ds.map(
            lambda x: tokenize_for_inference(x, tokenizer),
            batched=True,
            remove_columns=cols_to_remove
        )
        
        # Get predictions
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        predictions = trainer.predict(test_ds_tok)
        
        # Get predictions and probabilities for confidence scoring
        logits = predictions.predictions
        
        # Compute softmax probabilities for confidence scores
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        
        pred_labels = np.argmax(logits, axis=2)
        
        # Store probabilities for later filtering
        pred_probs = np.max(probs, axis=2)
        
        # Convert to markers with confidence tracking
        for i in range(len(pred_labels)):
            pred_ids = pred_labels[i]
            probs_seq = pred_probs[i]
            offsets = test_ds_tok[i]['offset_mapping']
            original_text = test_data_original[i]['text']
            
            markers = []
            current_span_start_char = None
            current_span_start_idx = None
            
            # Iterate over tokens
            for token_idx, label_id in enumerate(pred_ids):
                offset_tuple = offsets[token_idx]
                
                # Check for special tokens, padding, or tokens outside the text range
                is_special = (offset_tuple is None or 
                            offset_tuple[0] is None or 
                            offset_tuple[1] is None or 
                            (offset_tuple[0] == 0 and offset_tuple[1] == 0))
                
                if is_special:
                    # If we were tracking a span, close it using the end of the *previous* non-special token
                    if current_span_start_char is not None:
                        prev_end_char = None
                        # Find the end of the last valid token
                        if token_idx > 0 and offsets[token_idx - 1][1] is not None:
                            prev_end_char = offsets[token_idx - 1][1]
                        
                        if prev_end_char is not None:
                            # Calculate average confidence for this span
                            span_confidence = float(np.mean(probs_seq[current_span_start_idx:token_idx]))
                            markers.append({
                                'startIndex': int(current_span_start_char),
                                'endIndex': int(prev_end_char),
                                'type': marker_type,
                                'confidence': span_confidence
                            })
                        
                        current_span_start_char = None
                        current_span_start_idx = None
                    continue
                
                label = id_to_label[label_id]
                start_char = offset_tuple[0]
                
                if label == marker_type:
                    # Start or continue a span
                    if current_span_start_char is None:
                        # Start new span
                        current_span_start_char = start_char
                        current_span_start_idx = token_idx
                
                elif label == 'O':
                    # End the span if one was active
                    if current_span_start_char is not None:
                        # End is the end of the PREVIOUS token. The token at current_idx is 'O'.
                        # We need the end of the token at token_idx - 1.
                        prev_end_char = offsets[token_idx - 1][1] if token_idx > 0 and offsets[token_idx - 1][1] is not None else start_char
                        
                        # Calculate average confidence for this span
                        span_confidence = float(np.mean(probs_seq[current_span_start_idx:token_idx]))
                        markers.append({
                            'startIndex': int(current_span_start_char),
                            'endIndex': int(prev_end_char),
                            'type': marker_type,
                            'confidence': span_confidence
                        })
                        current_span_start_char = None
                        current_span_start_idx = None
            
            # After loop: Finalize any span that was still open at the end of the sequence
            if current_span_start_char is not None:
                # The end of the span is the end of the last non-special token in the sequence
                last_valid_end = None
                last_token_idx = len(pred_ids) - 1
                # Search backwards from the end of the sequence for the last non-special token's end index
                while last_token_idx >= 0:
                    offset_tuple_end = offsets[last_token_idx]
                    # Check if the token at this index is not a special token
                    if offset_tuple_end is not None and offset_tuple_end[1] is not None and offset_tuple_end[1] != 0:
                        last_valid_end = offset_tuple_end[1]
                        break
                    last_token_idx -= 1
                
                if last_valid_end is not None:
                    # Calculate average confidence for this span
                    span_confidence = float(np.mean(probs_seq[current_span_start_idx:last_token_idx+1]))
                    markers.append({
                        'startIndex': int(current_span_start_char),
                        'endIndex': int(last_valid_end),
                        'type': marker_type,
                        'confidence': span_confidence
                    })
            
            all_predictions[marker_type].append(markers)
        
        count = sum(len(m) for m in all_predictions[marker_type])
        print(f'  ✓ Found {count} {marker_type} markers')
    
    # Combine predictions - keep ONLY the best (highest confidence) span per marker type
    print('\nCombining predictions from 5 models...')
    print('  Strategy: Keep only the BEST span per marker type (highest confidence)')
    combined_markers = []
    
    for i in range(len(test_data_original)):
        example_markers = []
        
        # For each marker type, keep only the highest confidence prediction
        for marker_type in MARKER_TYPES:
            if i < len(all_predictions[marker_type]) and all_predictions[marker_type][i]:
                type_predictions = all_predictions[marker_type][i]
                if type_predictions:
                    # Sort by confidence and take the best one
                    best_prediction = max(type_predictions, key=lambda x: x['confidence'])
                    # Remove confidence before adding to output
                    example_markers.append({
                        'startIndex': best_prediction['startIndex'],
                        'endIndex': best_prediction['endIndex'],
                        'type': best_prediction['type']
                    })
        
        # Sort by start index for cleaner output
        example_markers = sorted(example_markers, key=lambda x: x['startIndex'])
        combined_markers.append(example_markers)
    
    total_markers = sum(len(m) for m in combined_markers)
    print(f'✓ Combined {total_markers} markers across {len(combined_markers)} examples')
    
    print('\nBreakdown by type:')
    for marker_type in MARKER_TYPES:
        count = sum(len(all_predictions[marker_type][i]) for i in range(len(test_data_original)))
        print(f'  {marker_type}: {count}')
    
    # Create submission files
    MARKERS_DIR = BASE / 'Markers-Extraction'
    SUBMISSION_FILE = MARKERS_DIR / 'submission_5models.jsonl'
    SUBMISSION_ZIP = MARKERS_DIR / 'submission_5models.zip'
    
    print(f'\nCreating submission file: {SUBMISSION_FILE}')
    with open(SUBMISSION_FILE, 'w') as f:
        for i, example in enumerate(test_data_original):
            submission_entry = {
                '_id': example['_id'],
                'markers': combined_markers[i]
            }
            f.write(json.dumps(submission_entry) + '\n')
    
    print(f'✅ Submission file created')
    
    # Create ZIP
    print(f'\nCreating ZIP: {SUBMISSION_ZIP}')
    with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(SUBMISSION_FILE, arcname='submission.jsonl')
    
    print(f'✅ ZIP file created')
    
    print('\n' + '='*70)
    print('🎉 SUBMISSION READY (5-MODEL APPROACH)!')
    print('='*70)
    print(f'📦 File to upload: {SUBMISSION_ZIP}')
    print(f'🌐 Upload to: https://www.codabench.org/competitions/10751/')
    print('='*70)
    
    # Show sample predictions
    print('\n' + '='*70)
    print('SAMPLE PREDICTIONS')
    print('='*70)
    
    for i in range(min(3, len(test_data_original))):
        example = test_data_original[i]
        markers = combined_markers[i]
        
        print(f'\n📄 Example {i+1} (ID: {example["_id"]})')
        print(f'Text: {example["text"][:100]}...')
        print(f'Predicted {len(markers)} markers:')
        
        for marker in markers[:10]:
            text_snippet = example["text"][marker["startIndex"]:marker["endIndex"]]
            print(f'  • {marker["type"]}: "{text_snippet}" ({marker["startIndex"]}-{marker["endIndex"]})')
        
        if len(markers) > 10:
            print(f'  ... and {len(markers) - 10} more markers')

if __name__ == '__main__':
    main()
