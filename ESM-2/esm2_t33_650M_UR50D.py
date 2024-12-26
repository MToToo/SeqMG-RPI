import torch
import esm
import os
import numpy as np
import sys
from tqdm import tqdm


def read_fasta(filename):
    with open(filename, 'r') as file:
        sequences = {}
        name = None
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:]  # Remove '>' symbol
                sequences[name] = ""
            else:
                sequences[name] += line
    return sequences


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python ./ESM-2/esm2_t33_650M_UR50D.py <protein_seq_fasta_file>")
        sys.exit(1)

    # Get command line arguments
    fasta_file = sys.argv[1]
    device_index = sys.argv[2] if len(sys.argv) > 2 else '0'
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000  # Default window size 1000
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 500  # Default stride 500
    print(f"window size: {window_size}, stride: {stride}")

    # Read protein sequences from the .fa file
    sequences = read_fasta(fasta_file)

    # Base output directory
    base_output_dir = os.path.splitext(os.path.basename(fasta_file))[0]
    output_dir = f"./ESM-2/esm2-{base_output_dir}_w{window_size}_s{stride}"

    representation_dir = os.path.join(output_dir, "representations")
    contact_map_dir = os.path.join(output_dir, "contact_maps")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(representation_dir, exist_ok=True)
    os.makedirs(contact_map_dir, exist_ok=True)

    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    # Split protein sequence into subsequences
    def split_sequence(seq, window_size, stride):
        subseqs = []
        seq_len = len(seq)

        # Force overwrite the beginning part
        subseqs.append(seq[:window_size])

        # Sliding window segmentation
        for start in range(0, seq_len - window_size, stride):
            subseqs.append(seq[start:start + window_size])

        # Force overwrite the end part
        if seq_len > window_size:
            subseqs.append(seq[-window_size:])

        return subseqs

    # Store skipped protein sequence names
    skipped_sequences = []

    # Process each protein sequence
    for name, seq in tqdm(sequences.items()):
        try:
            seq_len = len(seq)

            if seq_len <= window_size:
                subseqs = [seq]
            else:
                subseqs = split_sequence(seq, window_size=window_size, stride=stride)

            # Initialize full contact map and representation
            full_contact_map = np.zeros((seq_len, seq_len))
            full_representation = np.zeros((seq_len, 1280))
            count_map = np.zeros(seq_len)

            start_idx = 0

            # Make predictions for each segmented subsequence
            for i, subseq in enumerate(subseqs):
                batch_data = [(name, subseq)]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                # Use the model for inference
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

                token_representations = results["representations"][33].cpu()
                contact_map = results["contacts"][0, :batch_lens[0], :batch_lens[0]].cpu().numpy()

                # Extract sequence representation and remove special markers at the beginning and end
                sequence_representation = token_representations[0, 1:batch_lens[0] - 1].numpy()

                # Process the beginning and end parts to avoid overlap
                end_idx = min(start_idx + sequence_representation.shape[0], seq_len)

                if i == 0:
                    # Process the beginning part: directly place it
                    full_representation[start_idx:end_idx] = sequence_representation[:end_idx - start_idx]
                elif i == len(subseqs) - 1:
                    # Process the end part: directly place it
                    full_representation[start_idx:] = sequence_representation[:seq_len - start_idx]
                else:
                    # For the middle part, take the non-overlapping portion
                    overlap_start = stride
                    full_representation[start_idx + overlap_start:end_idx] = sequence_representation[
                                                                             overlap_start:end_idx - start_idx]

                # Merge the contact map into the full contact map
                full_contact_map[start_idx:end_idx, start_idx:end_idx] += contact_map[:end_idx - start_idx,
                                                                          :end_idx - start_idx]

                count_map[start_idx:end_idx] += 1

                start_idx += stride

                del batch_tokens
                del results
                torch.cuda.empty_cache()

            # Average overlapping regions
            for i in range(seq_len):
                if count_map[i] > 0:
                    full_contact_map[i, :] /= count_map[i]
                    full_contact_map[:, i] /= count_map[i]

            # Check if filling is complete
            zero_rows = np.where(np.sum(full_contact_map, axis=1) == 0)[0]
            if len(zero_rows) > 0:
                print(
                    f"Warning: Sequence '{name}' has {len(zero_rows)} rows in contact map with all zeros. Indices: {zero_rows}")

            # Save the representations and contact maps
            representation_file = os.path.join(representation_dir, f"{name}_representations.npy")
            np.save(representation_file, full_representation)

            contact_map_file = os.path.join(contact_map_dir, f"{name}_contact_map.npy")
            np.save(contact_map_file, full_contact_map)

        except torch.cuda.OutOfMemoryError:
            skipped_sequences.append(name)
            print(f"Sequence '{name}' caused CUDA out of memory error and was skipped.")

    # Print skipped protein sequence names due to insufficient CUDA memory
    if not skipped_sequences:
        print("All protein sequences have been processed successfully.")
        print("connect map file and representations file have been save to" + output_dir)
    else:
        print("Proteins skipped due to insufficient CUDA memory:")
        for skipped_seq in skipped_sequences:
            print(skipped_seq)
        print("The remaining protein sequences have been processed")


if __name__ == "__main__":
    main()
