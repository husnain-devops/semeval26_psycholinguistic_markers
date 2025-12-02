import jsonlines
import pandas as pd

rows = []
with jsonlines.open("train_redacted.jsonl") as reader:
    for obj in reader:
        rows.append(obj)

df = pd.DataFrame(rows)
df.shape
