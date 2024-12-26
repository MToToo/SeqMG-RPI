import torch
import torch.nn as nn
import torch.nn.functional as F
from src.protein_feature_extraction import ProteinFeatureExtractor
from src.rna_feature_extraction import RNAFeatureExtractor, RNAFeatureExtractorKmer

class InteractionPredictionModel(nn.Module):
    def __init__(self, rna_input_dim, rna_output_dim,
                 rna_kmer_input_dim, rna_kmer_output_dim, rna_svd_dim,
                 protein_input_dim, protein_hidden_dim, protein_output_dim, final_hidden_dim,
                 num_layers=2, nhead=8, transformer_hidden_dim=512):
        super(InteractionPredictionModel, self).__init__()

        # RNA and protein feature extractors
        self.rna_extractor = RNAFeatureExtractor(rna_input_dim, rna_output_dim)
        self.rna_kmer_extractor = RNAFeatureExtractorKmer(rna_kmer_input_dim, rna_kmer_output_dim)
        self.protein_extractor = ProteinFeatureExtractor(protein_input_dim, protein_hidden_dim, protein_output_dim)

        # Feature dimension after concatenation
        combined_input_dim = rna_output_dim + rna_kmer_output_dim + rna_svd_dim + protein_output_dim

        self.fc1 = nn.Linear(combined_input_dim, final_hidden_dim*2)
        self.ln1 = nn.LayerNorm(final_hidden_dim*2)

        self.fc2 = nn.Linear(final_hidden_dim*2, final_hidden_dim)
        self.ln2 = nn.LayerNorm(final_hidden_dim)

        self.fc3 = nn.Linear(final_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rna_features, rna_kmer, rna_svd, protein_batch):
        # Extract RNA, k-mer, and protein features
        rna_vector = self.rna_extractor(rna_features)
        rna_kmer_vector = self.rna_kmer_extractor(rna_kmer)
        rna_svd_vector = rna_svd
        protein_vector = self.protein_extractor(protein_batch.x, protein_batch.edge_index, protein_batch.edge_attr,
                                                protein_batch.batch)

        # Concatenate feature vectors
        combined = torch.cat((rna_vector, rna_kmer_vector, rna_svd_vector, protein_vector), dim=1)  # 形状为 [batch_size, combined_input_dim]

        x = F.relu(self.ln1(self.fc1(combined)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

