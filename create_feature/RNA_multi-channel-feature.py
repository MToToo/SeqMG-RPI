import os
import random
import numpy as np
import sys


def rna_to_kmer_matrix(rna_sequence):
    # Initialize a 4x8x8 matrix with zeros
    matrix = np.zeros((4, 8, 8), dtype=int)

    # Mapping for nucleotides to indices
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    # Iterate through the RNA sequence to form 4-mers
    # print("4mer")
    for i in range(len(rna_sequence) - 4 + 1):
        kmer = rna_sequence[i:i + 4]

        if len(kmer) == 4:
            first_nucleotide = kmer[0]
            second_nucleotide = kmer[1]
            third_nucleotide = kmer[2]
            fourth_nucleotide = kmer[3]

            if (first_nucleotide in nucleotide_to_index and
                    second_nucleotide in nucleotide_to_index and
                    third_nucleotide in nucleotide_to_index and
                    fourth_nucleotide in nucleotide_to_index):
                # Determine the channel
                channel = nucleotide_to_index[first_nucleotide]

                # Determine the sub-matrix position
                sub_matrix_row = nucleotide_to_index[third_nucleotide]
                sub_matrix_col = nucleotide_to_index[fourth_nucleotide]

                # Increment the corresponding position
                # matrix[channel, sub_matrix_row, sub_matrix_col] += 1

                if second_nucleotide == 'A':
                    matrix[channel, sub_matrix_row, sub_matrix_col] += 1
                elif second_nucleotide == 'C':
                    matrix[channel, sub_matrix_row, sub_matrix_col + 4] += 1
                elif second_nucleotide == 'G':
                    matrix[channel, sub_matrix_row + 4, sub_matrix_col] += 1
                elif second_nucleotide == 'U':
                    matrix[channel, sub_matrix_row + 4, sub_matrix_col + 4] += 1

    # 3-mer
    # print("3mer")
    for i in range(len(rna_sequence) - 3 + 1):
        kmer = rna_sequence[i:i + 3]

        if len(kmer) == 3:
            second_nucleotide = kmer[0]
            third_nucleotide = kmer[1]
            fourth_nucleotide = kmer[2]

            if (second_nucleotide in nucleotide_to_index and
                    third_nucleotide in nucleotide_to_index and
                    fourth_nucleotide in nucleotide_to_index):

                sub_matrix_row = nucleotide_to_index[third_nucleotide]
                sub_matrix_col = nucleotide_to_index[fourth_nucleotide]

                if second_nucleotide == 'A':
                    matrix[0, sub_matrix_row, sub_matrix_col] += 1
                    matrix[1, sub_matrix_row, sub_matrix_col] += 1
                    matrix[2, sub_matrix_row, sub_matrix_col] += 1
                    matrix[3, sub_matrix_row, sub_matrix_col] += 1
                elif second_nucleotide == 'C':
                    matrix[0, sub_matrix_row, sub_matrix_col + 4] += 1
                    matrix[1, sub_matrix_row, sub_matrix_col + 4] += 1
                    matrix[2, sub_matrix_row, sub_matrix_col + 4] += 1
                    matrix[3, sub_matrix_row, sub_matrix_col + 4] += 1
                elif second_nucleotide == 'G':
                    matrix[0, sub_matrix_row + 4, sub_matrix_col] += 1
                    matrix[1, sub_matrix_row + 4, sub_matrix_col] += 1
                    matrix[2, sub_matrix_row + 4, sub_matrix_col] += 1
                    matrix[3, sub_matrix_row + 4, sub_matrix_col] += 1
                elif second_nucleotide == 'U':
                    matrix[0, sub_matrix_row + 4, sub_matrix_col + 4] += 1
                    matrix[1, sub_matrix_row + 4, sub_matrix_col + 4] += 1
                    matrix[2, sub_matrix_row + 4, sub_matrix_col + 4] += 1
                    matrix[3, sub_matrix_row + 4, sub_matrix_col + 4] += 1

    # 2-mer
    # print("2mer")
    for i in range(len(rna_sequence) - 2 + 1):
        kmer = rna_sequence[i:i + 2]

        if len(kmer) == 2:
            third_nucleotide = kmer[0]
            fourth_nucleotide = kmer[1]

            if (third_nucleotide in nucleotide_to_index and
                    fourth_nucleotide in nucleotide_to_index):
                sub_matrix_row = nucleotide_to_index[third_nucleotide]
                sub_matrix_col = nucleotide_to_index[fourth_nucleotide]

                matrix[0, sub_matrix_row, sub_matrix_col] += 1
                matrix[0, sub_matrix_row, sub_matrix_col + 4] += 1
                matrix[0, sub_matrix_row + 4, sub_matrix_col] += 1
                matrix[0, sub_matrix_row + 4, sub_matrix_col + 4] += 1

                matrix[1, sub_matrix_row, sub_matrix_col] += 1
                matrix[1, sub_matrix_row, sub_matrix_col + 4] += 1
                matrix[1, sub_matrix_row + 4, sub_matrix_col] += 1
                matrix[1, sub_matrix_row + 4, sub_matrix_col + 4] += 1

                matrix[2, sub_matrix_row, sub_matrix_col] += 1
                matrix[2, sub_matrix_row, sub_matrix_col + 4] += 1
                matrix[2, sub_matrix_row + 4, sub_matrix_col] += 1
                matrix[2, sub_matrix_row + 4, sub_matrix_col + 4] += 1

                matrix[3, sub_matrix_row, sub_matrix_col] += 1
                matrix[3, sub_matrix_row, sub_matrix_col + 4] += 1
                matrix[3, sub_matrix_row + 4, sub_matrix_col] += 1
                matrix[3, sub_matrix_row + 4, sub_matrix_col + 4] += 1

    return matrix


def replace_invalid_nucleotides(sequence, sequence_name):
    valid_nucleotides = ['A', 'C', 'G', 'U']
    new_sequence = []

    # Replace all 'T' with 'U'
    sequence = sequence.replace('T', 'U')

    for i, n in enumerate(sequence):
        if n not in valid_nucleotides:
            replacement = random.choice(valid_nucleotides)
            print(f"RNA({sequence_name}): {n} at position {i + 1} is randomly replaced by {replacement}")
            new_sequence.append(replacement)
        else:
            new_sequence.append(n)
    return ''.join(new_sequence)


def process_sequences(input_filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_filename, 'r') as infile:
        sequence_name = None
        sequence = ''
        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                if sequence_name is not None:
                    sequence = replace_invalid_nucleotides(sequence, sequence_name)
                    new_rna_matrix = rna_to_kmer_matrix(sequence)
                    output_filename = os.path.join(output_dir, sequence_name[1:] + ".npy")
                    np.save(output_filename, new_rna_matrix)
                sequence_name = line
                sequence = ''
            else:
                sequence += line.upper()

        if sequence_name is not None:
            sequence = replace_invalid_nucleotides(sequence, sequence_name)
            new_rna_matrix = rna_to_kmer_matrix(sequence)
            output_filename = os.path.join(output_dir, sequence_name[1:] + ".npy")
            np.save(output_filename, new_rna_matrix)

    print("Multi-channel features of all RNA sequences have been generated and saved to" + output_dir)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ./create_feature/RNA_multi-channel-feature.py <rna_seq_fasta_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_dir = "./features/RNA_multi-channel-features"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    process_sequences(input_filename, output_dir)


if __name__ == "__main__":
    main()
