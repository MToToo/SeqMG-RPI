import csv
import os
import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from src.data_loader import load_data, collate_fn
from src.train import train_model, evaluate_model
from src.model import InteractionPredictionModel
from src.utils import load_pairs
import sys

def main():
    rna_mutil_channel_feature_dir = "./features/RNA_multi-channel-features"
    rna_kmer_frequency_feature_dir = "./features/RNA_kmer-frequency-features"
    rna_sparse_matrix_feature_dir = "./features/RNA_sparse-matrix-features"
    protein_graph_feature_dir = "./features/protein_graph-feature"

    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset_pair_file>")
        sys.exit(1)

    pairs_file = sys.argv[1]

    device_index = sys.argv[2] if len(sys.argv) > 2 else '0'
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pairs_name = os.path.splitext(os.path.basename(pairs_file))[0]
    print(f"paris_file: {pairs_name}")

    pairs = load_pairs(pairs_file)

    rna_input_channels = 4
    rna_output_dim = 128
    rna_kmer_input_dim = 64
    rna_kmer_output_dim = 64
    rna_svd_dim = 256
    protein_input_dim = 1280
    protein_hidden_dim = 256
    protein_output_dim = 512
    final_hidden_dim = 64

    dataset = load_data(rna_mutil_channel_feature_dir,
                        rna_kmer_frequency_feature_dir,
                        rna_sparse_matrix_feature_dir,
                        protein_graph_feature_dir,
                        pairs)

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_val_losses = []
    all_val_metrics = {
        'Acc': [],
        'Sens': [],
        'Spec': [],
        'Pre': [],
        'MCC': [],
        'AUC': [],
        'F1': [],
        'AUPR': []
    }

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        if len(train_subset) == 0 or len(val_subset) == 0:
            print(f"Empty subset in fold {fold + 1}")
            continue

        print(f"Fold {fold + 1} - Train set size: {len(train_subset)}, Val set size: {len(val_subset)}")

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=True)

        num_layers = 2
        model = InteractionPredictionModel(rna_input_channels, rna_output_dim,
                                           rna_kmer_input_dim, rna_kmer_output_dim, rna_svd_dim,
                                           protein_input_dim, protein_hidden_dim, protein_output_dim,
                                           final_hidden_dim, num_layers).to(device)

        best_model = train_model(model, train_loader, val_loader, num_epochs=1000, learning_rate=0.00001, device=device)

        val_loss, val_metrics, val_labels, val_probs, val_ids = evaluate_model(best_model, val_loader, nn.BCELoss(), device)
        print(f"Validation Loss: {val_loss}, Validation Metrics: {val_metrics}")

        # Collect the validation set results of this fold (protein_id, rna_id, true_label, pre_interaction_scores)
        results = []
        for idx, (protein_id, rna_id) in enumerate(val_ids):
            true_label = val_labels[idx]
            pred_prob = val_probs[idx]
            results.append([protein_id, rna_id, true_label, pred_prob])

        # Save the results of the current fold to a CSV file
        output_csv = f"results/{pairs_name}/RPI_pred-prob_fold_{fold + 1}.csv"

        output_dir = os.path.dirname(output_csv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_csv, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Protein_ID", "RNA_ID", "True_Label", "Predicted_Label"])
            csvwriter.writerows(results)

        print(f"Results for Fold {fold + 1} saved to {output_csv}")

        all_val_losses.append(val_loss)
        for key in all_val_metrics:
            all_val_metrics[key].append(val_metrics[key])

    avg_val_loss = sum(all_val_losses) / len(all_val_losses)
    avg_val_metrics = {key: sum(values) / len(values) for key, values in all_val_metrics.items()}

    print(f"Average Validation Loss: {avg_val_loss}")
    print(f"Average Validation Acc: {avg_val_metrics['Acc']}, Sens: {avg_val_metrics['Sens']}, "
          f"Spec: {avg_val_metrics['Spec']}, Pre: {avg_val_metrics['Pre']}, "
          f"MCC: {avg_val_metrics['MCC']}, AUC: {avg_val_metrics['AUC']}, "
          f"F1: {avg_val_metrics['F1']}, AUPR: {avg_val_metrics['AUPR']}")


if __name__ == '__main__':
    main()
