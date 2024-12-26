import os
import sys
import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from src.data_loader import load_data, collate_fn
from src.train import train_model, evaluate_model
from src.model import InteractionPredictionModel
from src.utils import load_pairs
import csv

def main():
    rna_mutil_channel_feature_dir = "./features/RNA_multi-channel-features"
    rna_kmer_frequency_feature_dir = "./features/RNA_kmer-frequency-features"
    rna_sparse_matrix_feature_dir = "./features/RNA_sparse-matrix-features"
    protein_graph_feature_dir = "./features/protein_graph-feature"

    if len(sys.argv) < 3:
        print("Usage: python main2.py <train_dataset_pair_file> <test_dataset_pair_file>")
        sys.exit(1)

    train_pairs_file = sys.argv[1]
    test_pairs_file = sys.argv[2]

    device_index = sys.argv[3] if len(sys.argv) > 3 else '0'
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_pairs_name = os.path.splitext(os.path.basename(train_pairs_file))[0]
    print(f"train_paris_file: {train_pairs_name}")
    test_pairs_name = os.path.splitext(os.path.basename(test_pairs_file))[0]
    print(f"test_paris_file: {test_pairs_name}")

    train_pairs = load_pairs(train_pairs_file)
    test_pairs = load_pairs(test_pairs_file)

    rna_input_channels = 4
    rna_output_dim = 128
    rna_kmer_input_dim = 64
    rna_kmer_output_dim = 64
    rna_svd_dim = 256
    protein_input_dim = 1280
    protein_hidden_dim = 256
    protein_output_dim = 512
    final_hidden_dim = 64

    train_dataset = load_data(rna_mutil_channel_feature_dir,
                              rna_kmer_frequency_feature_dir,
                              rna_sparse_matrix_feature_dir,
                              protein_graph_feature_dir,
                              train_pairs)

    test_dataset = load_data(rna_mutil_channel_feature_dir,
                             rna_kmer_frequency_feature_dir,
                             rna_sparse_matrix_feature_dir,
                             protein_graph_feature_dir,
                             test_pairs)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_test_losses = []
    all_test_metrics = {
        'Acc': [],
        'Sens': [],
        'Spec': [],
        'Pre': [],
        'MCC': [],
        'AUC': [],
        'F1': [],
        'AUPR': []
    }

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

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

        test_loss, test_metrics, test_labels, test_probs, test_ids = evaluate_model(best_model, test_loader,nn.BCELoss(), device)
        print(f"Validation Loss: {test_loss}, Validation Metrics: {test_metrics}")

        test_acc_value = test_metrics['Acc'] * 100
        test_acc_value = round(test_acc_value, 2)

        # Collect the validation set results of this fold (protein_id, rna_id, true_label, pre_interaction_scores)
        results = []
        for idx, (protein_id, rna_id) in enumerate(test_ids):
            true_label = test_labels[idx]
            pred_prob = test_probs[idx]
            results.append([protein_id, rna_id, true_label, pred_prob])

        # Save the results of the current fold to a CSV file
        output_csv = f"result/T-{train_pairs_name}_t-{test_pairs_name}/fold{fold + 1}-acc{test_acc_value}.csv"

        output_dir = os.path.dirname(output_csv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_csv, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Protein_ID", "RNA_ID", "True_Label", "Predicted_Label"])
            csvwriter.writerows(results)

        print(f"Results for Fold {fold + 1} saved to {output_csv}")

        all_test_losses.append(test_loss)
        for key in all_test_metrics:
            all_test_metrics[key].append(test_metrics[key])

    avg_test_loss = sum(all_test_losses) / len(all_test_losses)
    # avg_test_acc = sum(all_test_metrics['Acc']) / len(all_test_metrics['Acc'])
    avg_test_metrics = {}
    for key, values in all_test_metrics.items():
        non_none_values = [val for val in values if val is not None]
        if len(non_none_values) == 0:
            avg_test_metrics[key] = None
        else:
            avg_test_metrics[key] = sum(non_none_values) / len(non_none_values)

    print(f"train_paris_file: {train_pairs_name}")
    print(f"test_paris_file: {test_pairs_name}")
    print(f"Average Validation Loss: {avg_test_loss}")
    # print(f"Average Validation Acc: {avg_test_acc}")
    print(f"Average Validation Acc: {avg_test_metrics['Acc']}, Sens: {avg_test_metrics['Sens']}, "
          f"Spec: {avg_test_metrics['Spec']}, Pre: {avg_test_metrics['Pre']}, "
          f"MCC: {avg_test_metrics['MCC']}, AUC: {avg_test_metrics['AUC']}, "
          f"F1: {avg_test_metrics['F1']}, AUPR: {avg_test_metrics['AUPR']}")

if __name__ == '__main__':
    main()
