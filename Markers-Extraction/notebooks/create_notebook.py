import json

# Create a comprehensive RoBERTa-LoRA notebook for marker extraction
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Add cells
cells = [
    # Cell 0: Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Psycholinguistic Marker Extraction with RoBERTa-LoRA\n",
            "## Token Classification for Conspiracy Marker Detection\n",
            "\n",
            "This notebook implements marker extraction using RoBERTa with LoRA:\n",
            "- **Task**: Token classification (NER) for psycholinguistic markers\n",
            "- **Markers**: Action, Actor, Effect, Evidence, Victim\n",
            "- **Model**: RoBERTa-base with LoRA fine-tuning\n",
            "- **Tagging**: BIO scheme (Beginning-Inside-Outside)\n"
        ]
    },
    # Cell 1: Environment check
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
            "\n",
            "import torch\n",
            "import numpy as np\n",
            "\n",
            "print('='*60)\n",
            "print('Environment Check')\n",
            "print('='*60)\n",
            "print(f'PyTorch: {torch.__version__}')\n",
            "print(f'NumPy: {np.__version__}')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            "else:\n",
            "    print('Device: CPU')\n",
            "print('='*60)"
        ]
    },
    # Cell 2: Imports
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "from pathlib import Path\n",
            "from transformers import (\n",
            "    RobertaTokenizerFast,\n",
            "    RobertaForTokenClassification,\n",
            "    TrainingArguments,\n",
            "    Trainer,\n",
            "    DataCollatorForTokenClassification,\n",
            "    EarlyStoppingCallback\n",
            ")\n",
            "from peft import LoraConfig, get_peft_model, TaskType\n",
            "from datasets import Dataset\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import classification_report, f1_score\n",
            "\n",
            "# Set seeds\n",
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.manual_seed_all(42)\n",
            "\n",
            "print('✓ Libraries imported')"
        ]
    },
    # Cell 3: Load data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Paths\n",
            "BASE = Path('../..')\n",
            "TRAIN_FILE = BASE / 'train_rehydrated.jsonl'\n",
            "DATA_DIR = Path('../data_processed')\n",
            "DATA_DIR.mkdir(exist_ok=True)\n",
            "\n",
            "def load_data(file_path):\n",
            "    data = []\n",
            "    with open(file_path) as f:\n",
            "        for line in f:\n",
            "            try:\n",
            "                data.append(json.loads(line))\n",
            "            except:\n",
            "                pass\n",
            "    return data\n",
            "\n",
            "print('Loading data...')\n",
            "train_data = load_data(TRAIN_FILE)\n",
            "print(f'✓ Loaded {len(train_data)} examples')\n",
            "\n",
            "# Show sample\n",
            "sample = train_data[0]\n",
            "print(f'\\nSample text: {sample[\"text\"][:150]}...')\n",
            "print(f'Markers: {len(sample.get(\"markers\", []))}')\n",
            "for m in sample.get('markers', [])[:3]:\n",
            "    print(f'  - {m[\"type\"]}: \"{m[\"text\"]}\"')"
        ]
    },
    # Cell 4: Label mapping
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# BIO tagging scheme\n",
            "MARKER_TYPES = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']\n",
            "\n",
            "label_list = ['O']  # Outside\n",
            "for mt in MARKER_TYPES:\n",
            "    label_list.extend([f'B-{mt}', f'I-{mt}'])\n",
            "\n",
            "label_to_id = {l: i for i, l in enumerate(label_list)}\n",
            "id_to_label = {i: l for l, i in label_to_id.items()}\n",
            "num_labels = len(label_list)\n",
            "\n",
            "print('='*60)\n",
            "print('Label Mapping (BIO Tagging)')\n",
            "print('='*60)\n",
            "print(f'Total labels: {num_labels}')\n",
            "print(f'Marker types: {MARKER_TYPES}')\n",
            "print(f'Labels: {label_list[:5]}... (showing first 5)')\n",
            "print('='*60)"
        ]
    },
    # Cell 5: Tokenization
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "MODEL_NAME = 'roberta-base'\n",
            "MAX_LENGTH = 256\n",
            "\n",
            "tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)\n",
            "\n",
            "def tokenize_and_align_labels(examples):\n",
            "    tokenized_inputs = tokenizer(\n",
            "        examples['text'],\n",
            "        truncation=True,\n",
            "        max_length=MAX_LENGTH,\n",
            "        return_offsets_mapping=True,\n",
            "        is_split_into_words=False\n",
            "    )\n",
            "    \n",
            "    labels = []\n",
            "    all_markers = examples.get('markers', [])\n",
            "    \n",
            "    for i, offsets in enumerate(tokenized_inputs['offset_mapping']):\n",
            "        example_labels = [label_to_id['O']] * len(offsets)\n",
            "        example_markers = all_markers[i] if i < len(all_markers) else []\n",
            "        \n",
            "        example_markers = sorted(example_markers, key=lambda x: x['startIndex'])\n",
            "        \n",
            "        for marker in example_markers:\n",
            "            marker_type = marker['type']\n",
            "            start_char = marker['startIndex']\n",
            "            end_char = marker['endIndex']\n",
            "            \n",
            "            b_label = label_to_id.get(f'B-{marker_type}')\n",
            "            i_label = label_to_id.get(f'I-{marker_type}')\n",
            "            \n",
            "            if b_label is None or i_label is None:\n",
            "                continue\n",
            "            \n",
            "            first_token = True\n",
            "            for token_idx, (start, end) in enumerate(offsets):\n",
            "                if start is None or end is None:\n",
            "                    continue\n",
            "                \n",
            "                if start < end_char and end > start_char:\n",
            "                    if first_token:\n",
            "                        example_labels[token_idx] = b_label\n",
            "                        first_token = False\n",
            "                    else:\n",
            "                        if example_labels[token_idx] == label_to_id['O']:\n",
            "                            example_labels[token_idx] = i_label\n",
            "        \n",
            "        labels.append(example_labels)\n",
            "    \n",
            "    tokenized_inputs['labels'] = labels\n",
            "    return tokenized_inputs\n",
            "\n",
            "print(f'✓ Tokenizer: {MODEL_NAME}')\n",
            "print(f'✓ Max length: {MAX_LENGTH}')\n",
            "print(f'✓ BIO tagging configured')"
        ]
    },
    # Continue with more cells...
]

notebook["cells"] = cells

# Save notebook
with open('01_RoBERTa_LoRA_Markers_Extraction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook created successfully!")
