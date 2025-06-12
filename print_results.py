import csv
import os

RESULTS_ROOT_DIR="/path/to/results"

seq_lens = os.listdir(RESULTS_ROOT_DIR)
seq_lens = sorted(seq_lens)

for seq_len in seq_lens:
    seq_len_pred = os.path.join(RESULTS_ROOT_DIR, seq_len, "pred/summary.csv")
    if not os.path.exists(seq_len_pred):
        continue

    print(f"{seq_len}:")
    with open(seq_len_pred, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        tasks = next(reader)
        scores = next(reader)
        
        for task in tasks:
            print(f"{task}")
        print("==" * 20)
        for score in scores:
            print(f"{score}")
    print("\n")
    