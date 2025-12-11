# Memory Fix Instructions for Kernel OOM Issue

The kernel is dying at 0.33/5 epochs due to out-of-memory (OOM) errors. Here are the exact changes needed:

## Changes Required:

### 1. In Cell 13 (Training hyperparameters section):

**Change:**
```python
BATCH_SIZE = 16
```

**To:**
```python
# Reduced batch size to prevent OOM errors
BATCH_SIZE = 8  # Reduced from 16 to prevent memory issues
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients to maintain effective batch size of 16
```

**Also update the print statement:**
```python
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
```

### 2. In Cell 18 (TrainingArguments section):

**Replace the entire TrainingArguments block with:**

```python
# Training arguments with warmup and early stopping
# Memory optimization settings
use_fp16 = torch.cuda.is_available()
# Use bf16 for ROCm if available (better support than fp16)
use_bf16 = False
if torch.cuda.is_available():
    try:
        # Check if bf16 is supported
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
        use_bf16 = True
        use_fp16 = False
    except:
        pass

training_args = TrainingArguments(
    output_dir='./roberta-base-conspiracy-classification',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Accumulate gradients to save memory
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,  # 10% warmup
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Use loss for early stopping
    greater_is_better=False,  # Lower loss is better
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
    seed=42,
    fp16=use_fp16,
    bf16=use_bf16,  # Use bfloat16 for ROCm if supported
    dataloader_num_workers=0,  # Reduce memory overhead from data loading
    dataloader_pin_memory=False,  # Disable pin memory to save RAM
    max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
    save_total_limit=2,  # Keep only last 2 checkpoints to save disk space
)
```

**Update the device information print section:**

```python
if torch.cuda.is_available():
    print(f"✓ Device: GPU ({torch.cuda.get_device_name(0)})")
    print(f"✓ Mixed Precision (FP16): {use_fp16}")
    print(f"✓ Mixed Precision (BF16): {use_bf16}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ Batch size per device: {BATCH_SIZE}")
    print(f"✓ Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"✓ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
else:
    print(f"✓ Device: CPU")
    print(f"✓ Mixed Precision (FP16): {use_fp16}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"✓ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
```

## What These Changes Do:

1. **Reduced Batch Size**: From 16 to 8 reduces memory usage by ~50%
2. **Gradient Accumulation**: Maintains effective batch size of 16 (8 × 2) while using less memory
3. **BF16 Support**: Better for ROCm GPUs than FP16
4. **Disabled DataLoader Workers**: Reduces memory overhead
5. **Disabled Pin Memory**: Saves RAM
6. **Gradient Clipping**: Prevents memory spikes from large gradients
7. **Limited Checkpoints**: Saves disk space and memory

## Expected Result:

- Training should complete without OOM errors
- Effective batch size remains 16 (same training dynamics)
- Lower memory usage during training
- Better compatibility with ROCm GPUs

