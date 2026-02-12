import numpy as np
import pandas as pd
import random
from collections import Counter
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def consolidate_labels_tracked(example):
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

def get_preprocessed_data():

    ds = load_dataset("ailsntua/QEvasion")

    ds["train"] = ds["train"].class_encode_column("evasion_label")
    ds["test"]  = ds["test"].cast_column(
        "evasion_label",
        ds["train"].features["evasion_label"]
    )

    label_feature = ds["train"].features["evasion_label"]
    id2label = dict(enumerate(label_feature.names))
    label2id = {v: k for k, v in id2label.items()}

    sub_split = ds["train"].train_test_split(
        test_size=0.2,
        stratify_by_column="evasion_label",
        seed=SEED
    )

    test_set = ds["test"].map(consolidate_labels_tracked)

    train_df = sub_split["train"].to_pandas()
    val_df = sub_split["test"].to_pandas()
    test_df = test_set.to_pandas()

    return train_df, val_df, test_df, id2label, label2id

if __name__ == "__main__":
    train, val, test, i2l, l2i = get_preprocessed_data()
    print("Data loaded successfully.")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Labels: {i2l}")