import json
import torch
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from datasets import Dataset

BASE = Path(__file__).parent
TEST_FILE = BASE / 'test_rehydrated.jsonl'
MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']
MAX_LENGTH = 512

def run_inference():
    with open(TEST_FILE) as f:
        test_data = [json.loads(line) for line in f]
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    all_results = [[] for _ in range(len(test_data))]

    for m_type in MARKER_TYPES:
        print(f"Extracting {m_type}...")
        model_path = BASE / '../models' / f'roberta-{m_type}' / 'final'
        with open(model_path / 'config_labels.json') as f:
            id2l = {int(k): v for k, v in json.load(f)['id2l'].items()}
        
        #model = RobertaForTokenClassification.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        #model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RobertaForTokenClassification.from_pretrained(
        model_path, 
        num_labels=3, 
        ignore_mismatched_sizes=True
    ).to(device)

        for i, item in enumerate(test_data):
            inputs = tokenizer(item['text'], return_tensors="pt", truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True).to(model.device)
            offsets = inputs.pop('offset_mapping')[0].cpu().numpy()
            
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                preds = np.argmax(probs, axis=-1)

            curr_start = None
            for idx, label_id in enumerate(preds):
                label = id2l[label_id]
                o_start, o_end = offsets[idx]
                if o_start == o_end == 0: continue

                if label.startswith('B-'):
                    if curr_start is not None: # Close existing
                        all_results[i].append({'startIndex': int(curr_start), 'endIndex': int(prev_end), 'type': m_type})
                    curr_start = o_start
                    prev_end = o_end
                elif label.startswith('I-') and curr_start is not None:
                    prev_end = o_end
                else:
                    if curr_start is not None:
                        all_results[i].append({'startIndex': int(curr_start), 'endIndex': int(prev_end), 'type': m_type})
                        curr_start = None

    # Merge into original JSON structure
    # Merge into specific submission structure
    output_path = BASE / 'submission.jsonl'
    with open(output_path, 'w') as f:
        for i, original_item in enumerate(test_data):
            # Sort markers by startIndex for cleanliness
            markers = sorted(all_results[i], key=lambda x: x['startIndex'])
            
            # Create a clean dictionary with only the required fields
            output_item = {
                "_id": original_item['_id'],
                "markers": markers
            }
            
            f.write(json.dumps(output_item) + '\n')
    
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    run_inference()