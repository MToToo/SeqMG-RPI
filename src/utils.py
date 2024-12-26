import csv
import os
import numpy as np
import torch

# Loading RNA-protein pairs and labels
def load_pairs(pairs_file):
    pairs = []
    with open(pairs_file, 'r') as file:
        if pairs_file.endswith('.csv'):
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                protein_id, rna_id, label = row  # Read protein IDs, RNA IDs and labels from csv files
                pairs.append((protein_id, rna_id, int(label)))
        else:
            for line in file:
                protein_id, rna_id, label = line.strip().split()  # Read protein IDs, RNA IDs and labels from non-csv files
                pairs.append((protein_id, rna_id, int(label)))
    return pairs

# Calculate the maximum length of RNA feature vector
def find_max_rna_length(rna_dir):
    max_length = 0
    for filename in os.listdir(rna_dir):
        if filename.endswith(".npy"):
            rna_features = np.load(os.path.join(rna_dir, filename))
            if rna_features.shape[0] > max_length:
                max_length = rna_features.shape[0]
    return max_length

# Select GPU
def get_device(preferred_device_index=0):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > preferred_device_index:
            device = torch.device(f'cuda:{preferred_device_index}')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device
