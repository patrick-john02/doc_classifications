#This module will handle reading the index, locating images and preparing dataset records.
from pathlib import Path
import random 
from collections import defaultdict


INDEX_PATH = Path("data/raw/rvl_cdip/data/train.txt")

groups = defaultdict(list)
for line in INDEX_PATH.read_text().splitlines():
    image_path, label = line.rsplit(maxsplit=1)
    groups[int(label)].append(image_path)

rng = random.Random(42) #random.Random(42) — Uses a fixed random seed for reproducibility.
samples = [(path, label) for label, paths in groups.items()
           for path in rng.sample(paths, min(20, len(paths)))] #min() — Prevents requesting more samples than a class contains.
print(f"Selected {len(samples)} documents")
