import numpy as np
import pandas as pd
import gensim.downloader as api
import random
from collections import Counter

from datasets import load_dataset

def consolidate_labels_tracked(example):
    random.seed(123) 
    
    votes = [example['annotator1'], example['annotator2'], example['annotator3']]
    counts = Counter(votes)
    most_common = counts.most_common()
    
    if most_common[0][1] > 1:
        final_label = most_common[0][0]
        is_tie = False
    else:
        final_label = random.choice(votes)
        is_tie = True 
    
    return {'evasion_label': final_label, 'is_tie': is_tie}

SEED = 42
np.random.seed(SEED)

ds = load_dataset("ailsntua/QEvasion")

ds["train"] = ds["train"].class_encode_column("clarity_label")
ds["test"]  = ds["test"].cast_column(
    "clarity_label",
    ds["train"].features["clarity_label"]
)

label_feature = ds["train"].features["clarity_label"]
id2label = dict(enumerate(label_feature.names))
label2id = {v:k for k,v in id2label.items()}

print(id2label)

sub_split = ds["train"].train_test_split(
    test_size=0.2,
    stratify_by_column="clarity_label",
    seed=42
)

train_set = sub_split["train"]
val_set   = sub_split["test"]
test_set  = ds["test"]

train_df = train_set.to_pandas()
val_df = val_set.to_pandas()
test_df = test_set.to_pandas()

print(f"Train size: {len(train_df)}")
print(f"Val size:   {len(val_df)}")
print(f"Test size:  {len(test_df)}")

train_df.head()

test_set = test_set.map(consolidate_labels_tracked)

tie_count = sum(test_set['is_tie'])
total = len(test_set)

print(f"Total Rows: {total}")
print(f"Random Tie-Breakers Triggered: {tie_count}")
print(f"Percentage: {(tie_count/total)*100:.1f}%")