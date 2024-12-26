import sys
import numpy as np
from itertools import product
from sklearn.decomposition import TruncatedSVD
import os


# Generate all possible k-mers
def generate_full_kmer_vocab(k):
    bases = ['A', 'C', 'G', 'U']
    full_vocab = [''.join(p) for p in product(bases, repeat=k)]
    return full_vocab


# Constructing a sparse matrix
def build_kmer_sparse_matrix(rna_sequence, k=3):
    full_vocab = generate_full_kmer_vocab(k)
    vocab_index = {kmer: i for i, kmer in enumerate(full_vocab)}

    sequence_length = len(rna_sequence)
    
    if sequence_length <= k:
        # If the length of the RNA sequence is less than k, k-mer cannot be constructed and None is returned directly.
        return None, full_vocab

    num_kmers = sequence_length - k + 1
    kmer_matrix = np.zeros((len(full_vocab), num_kmers))

    for i in range(num_kmers):
        kmer = rna_sequence[i:i + k]
        if kmer in vocab_index:
            row = vocab_index[kmer]
            kmer_matrix[row, i] = 1

    return kmer_matrix, full_vocab


# SVD Dimensionality Reduction
def perform_svd_on_kmer_matrix(kmer_matrix):
    svd = TruncatedSVD(n_components=1)
    reduced_vector = svd.fit_transform(kmer_matrix)
    return reduced_vector.T


# Parsing FASTA files
def parse_fasta(fasta_file):
    rna_dict = {}
    with open(fasta_file, 'r') as file:
        current_rna_name = ""
        current_rna_sequence = []

        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_rna_name:
                    rna_dict[current_rna_name] = ''.join(current_rna_sequence)
                current_rna_name = line[1:]
                current_rna_sequence = []
            else:
                current_rna_sequence.append(line)

        if current_rna_name:
            rna_dict[current_rna_name] = ''.join(current_rna_sequence)

    return rna_dict


# Batch process RNA sequences and save the feature vectors after SVD dimensionality reduction
def process_rna_sequences(fasta_file, output_dir, k=4):
    rna_dict = parse_fasta(fasta_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rna_name, rna_sequence in rna_dict.items():
        # Constructing k-mer sparse matrix
        kmer_matrix, kmer_vocab = build_kmer_sparse_matrix(rna_sequence, k=k)

        if kmer_matrix is None:
            print(f"Warning: Sequence length of {rna_name} is less than or equal to {k}, skipping processing.")
            continue

        reduced_vector = perform_svd_on_kmer_matrix(kmer_matrix)

        output_file = os.path.join(output_dir, f"{rna_name}_svd.npy")
        np.save(output_file, reduced_vector)
    print("RNA sparse matrix features of all RNA sequences have been generated and saved to" + output_dir)


k = 4

def main():
    if len(sys.argv) < 2:
        print("Usage: python ./create_feature/RNA_sparse-matrix-feature.py <rna_seq_fasta_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_dir = "./features/RNA_sparse-matrix-features"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    process_rna_sequences(input_filename, output_dir, k)

if __name__ == "__main__":
    main()
