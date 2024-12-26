import os
import sys
import numpy as np
import pickle
import torch
import networkx as nx
from tqdm import *
from torch_geometric.data import Data

if len(sys.argv) < 2:
    print("Usage: python ./create_feature/protein_graph-feature.py <esm2_file_dir>")
    sys.exit(1)

esm2_file_dir = sys.argv[1]
output_dir = "./features/protein_graph-feature"

if len(sys.argv) > 2:
    output_dir = sys.argv[2]

representation_dir = f"{esm2_file_dir}/representations"
contact_map_dir = f"{esm2_file_dir}/contact_maps"
os.makedirs(output_dir, exist_ok=True)

representation_files = [f for f in os.listdir(representation_dir) if f.endswith('_representations.npy')]

# Set the contact threshold
contact_threshold = 0.5

# Convert NetworkX graph to PyTorch Geometric data object
def nx_to_tg_data(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous().long()  # 确保是 int64 类型
    node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_weights = np.array([data['weight'] for _, _, data in G.edges(data=True)])
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

for representation_file in tqdm(representation_files):
    protein_name = representation_file.replace('_representations.npy', '')
    contact_map_file = os.path.join(contact_map_dir, f"{protein_name}_contact_map.npy")

    if not os.path.exists(contact_map_file):
        print(f"Contact map file for {protein_name} not found.")
        continue

    representations = np.load(os.path.join(representation_dir, representation_file))
    contact_map = np.load(contact_map_file)

    G = nx.Graph()

    for i in range(representations.shape[0]):
        G.add_node(i, feature=representations[i])

    for i in range(contact_map.shape[0]):
        for j in range(contact_map.shape[1]):
            if i != j and contact_map[i, j] > contact_threshold:
                G.add_edge(i, j, weight=contact_map[i, j])

    data = nx_to_tg_data(G)

    data_file = os.path.join(output_dir, f"{protein_name}_data.pt")
    torch.save(data, data_file)

print(f"Converted and saved {len(representation_files)} graphs to {output_dir}.")
