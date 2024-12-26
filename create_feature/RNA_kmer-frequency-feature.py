import sys
import numpy as np
from collections import Counter
from itertools import product
import os


def get_kmer_frequencies(sequence, k):
    possible_kmers = [''.join(p) for p in product('ACGU', repeat=k)]
    kmer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1) if all(base in 'ACGU' for base in sequence[i:i+k])])

    kmer_frequencies = {kmer: 0 for kmer in possible_kmers}
    for kmer, count in kmer_counts.items():
        kmer_frequencies[kmer] = count / (len(sequence) - k + 1)

    feature_vector = np.array(list(kmer_frequencies.values()))
    return feature_vector.flatten()


def process_fasta(file_path, output_dir, k):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_name = None
    current_sequence = []

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_name is not None:
                sequence = ''.join(current_sequence)
                sequence = ''.join([base for base in sequence if base in 'ACGU'])  # Ignore nonstandard nucleotides
                if sequence:
                    feature_vector = get_kmer_frequencies(sequence, k)
                    np.save(os.path.join(output_dir, f'{current_name}_kmer.npy'), feature_vector)

            current_name = line[1:]
            current_sequence = []
        else:
            current_sequence.append(line)

        if current_name is not None:
            sequence = ''.join(current_sequence)
            sequence = ''.join([base for base in sequence if base in 'ACGU'])  # Ignore nonstandard nucleotides
            if sequence:
                feature_vector = get_kmer_frequencies(sequence, k)
                np.save(os.path.join(output_dir, f'{current_name}_kmer.npy'), feature_vector)
    print("RNA_kmer-frequency-features of all RNA sequences have been generated and saved to" + output_dir)


k = 3

def main():
    if len(sys.argv) < 2:
        print("Usage: python ./create_feature/RNA_kmer-frequency-feature.py <rna_seq_fasta_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_dir = "./features/RNA_kmer-frequency-features"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    process_fasta(input_filename, output_dir, k)

if __name__ == "__main__":
    main()
