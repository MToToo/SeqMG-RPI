import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class InteractionDataset(Dataset):
    def __init__(self, rna_dir, rna_kmer_dir, rna_svd_dir, protein_dir, pairs):
        self.rna_dir = rna_dir
        self.rna_kmer_dir = rna_kmer_dir
        self.rna_svd_dir = rna_svd_dir
        self.protein_dir = protein_dir
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        protein_id, rna_id, label = self.pairs[idx]

        # Load RNA multi channel features data
        rna_features = np.load(os.path.join(self.rna_dir, f"{rna_id}.npy"))
        rna_features = np.expand_dims(rna_features, axis=0)  # [1, 4, 8, 8]

        # Load RNA kmer frequency features data
        rna_kmer_features = np.load(os.path.join(self.rna_kmer_dir, f"{rna_id}_kmer.npy"))
        rna_kmer_features = torch.tensor(rna_kmer_features, dtype=torch.float32)

        # Load RNA sparse matrix features data
        rna_svd_features = np.load(os.path.join(self.rna_svd_dir, f"{rna_id}_svd.npy"))
        rna_svd_features = torch.tensor(rna_svd_features, dtype=torch.float32)  # (1,256)
        rna_svd_features = rna_svd_features.squeeze(0)  # Now the shape is (256,)

        # Load protein graph feature data
        protein_file = os.path.join(self.protein_dir, f"{protein_id}_data.pt")
        if not os.path.exists(protein_file):
            raise FileNotFoundError(f"Protein file not found: {protein_file}")
        protein_data = torch.load(protein_file)

        # Convert labels to torch tensors
        label = torch.tensor([label], dtype=torch.float32)

        return torch.tensor(rna_features, dtype=torch.float32), rna_kmer_features, rna_svd_features, protein_data, label, (protein_id, rna_id)

# Data Normalization
def min_max_normalize(data):
    min_vals = data.min(dim=0, keepdim=True)[0]
    max_vals = data.max(dim=0, keepdim=True)[0]
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-6)  # add small value to avoid division by zero
    return normalized_data


def collate_fn(batch):
    rna_features, rna_kmer, rna_svd, protein_data, labels, ids = zip(*batch)

    rna_features_batch = torch.stack(rna_features)
    rna_kmer_batch = torch.stack([min_max_normalize(kmer) for kmer in rna_kmer])
    rna_svd_batch = torch.stack(rna_svd)
    for i, protein in enumerate(protein_data):
        protein_data[i].x = min_max_normalize(protein.x)

    # Build batch data
    protein_batch = Batch.from_data_list(protein_data)
    labels = torch.stack(labels).squeeze()

    return rna_features_batch, rna_kmer_batch, rna_svd_batch, protein_batch, labels, ids


def load_data(rna_dir, rna_kmer_dir, rna_svd_dir, protein_dir, pairs):
    dataset = InteractionDataset(rna_dir, rna_kmer_dir, rna_svd_dir, protein_dir, pairs)
    return dataset
