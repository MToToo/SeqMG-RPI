# SeqMG-RPI: RNA-Protein Interaction Prediction Network

## 1. Introduction

SeqMG-RPI is an RNA-protein interaction prediction network based on RNA and protein sequences.

## 2. Operating System

SeqMG-RPI was developed on a Linux environment with CUDA 11.8.

Hardware: Two NVIDIA GeForce RTX 4090（24G）

## 3. Environment Setup

### Create and activate the environment

```bash
conda create -n seqmg python=3.8  # Create environment
conda activate seqmg  # Activate environment
```

### Install ESM-2 Model

Download the ESM-2 model and follow the [official tutorial](https://github.com/facebookresearch/esm) for installation.

```bash
pip install fair-esm  # latest release OR
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```

The ESM-2 model used in this project is `esm2_t33_650M_UR50D`. This model will be automatically downloaded during the first run of the code. If the download fails, please follow the [ESM-2 official tutorial](https://github.com/facebookresearch/esm) for manual download.

### Install Dependencies

```bash
conda install numpy  # numpy 1.24.3
conda install scikit-learn  # scikit-learn 1.3.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # pytorch 2.4.1
conda install tqdm  # tqdm 4.66.5
conda install pyg -c pyg  # torch_geometric 2.6.1
conda install pytorch-scatter -c pyg  # torch-scatter 2.1.2
```

## 4. Feature Generation

### 4.1 Generate Multi-Channel RNA Features

Run the following command to generate multi-channel RNA features for the dataset:

```bash
python ./create_feature/RNA_multi-channel-feature.py <rna_seq_fasta_file>  # <rna_seq_fasta_file> is the RNA sequence file path
```

For example, to generate multi-channel RNA features for the ATH948 dataset:

```bash
python ./create_feature/RNA_multi-channel-feature.py datasets/rna_seq/ATH948_rna_seq.fa
```

The output feature files will be stored by default in `./features/RNA_multi-channel-features`. To store them in a different location, use the following command to specify the output directory:

```bash
python ./create_feature/RNA_multi-channel-feature.py <rna_seq_fasta_file> <output_dir>  # <output_dir> is the directory where you want to store the feature files
```

### 4.2 Generate RNA k-mer Frequency Features

Run the following command to generate RNA k-mer frequency features for the dataset:

```bash
python ./create_feature/RNA_kmer-frequency-feature.py <rna_seq_fasta_file>  # <rna_seq_fasta_file> is the RNA sequence file path
```

For example, to generate RNA k-mer frequency features for the ATH948 dataset:

```bash
python ./create_feature/RNA_kmer-frequency-feature.py datasets/rna_seq/ATH948_rna_seq.fa
```

The output feature files will be stored by default in `./features/RNA_kmer-frequency-features`. To store them in a different location, use the following command to specify the output directory:

```bash
python ./create_feature/RNA_kmer-frequency-feature.py <rna_seq_fasta_file> <output_dir>  # <output_dir> is the directory where you want to store the feature files
```

### 4.3 Generate RNA Sparse Matrix Features

Run the following command to generate RNA sparse matrix features for the dataset:

```bash
python ./create_feature/RNA_sparse-matrix-feature.py <rna_seq_fasta_file>  # <rna_seq_fasta_file> is the RNA sequence file path
```

For example, to generate RNA sparse matrix features for the ATH948 dataset:

```bash
python ./create_feature/RNA_sparse-matrix-feature.py datasets/rna_seq/ATH948_rna_seq.fa
```

The output feature files will be stored by default in `./features/RNA_sparse-matrix-features`. To store them in a different location, use the following command to specify the output directory:

```bash
python ./create_feature/RNA_sparse-matrix-feature.py <rna_seq_fasta_file> <output_dir>  # <output_dir> is the directory where you want to store the feature files
```

### 4.4 Generate Protein Graph Features

#### 4.4.1 Use ESM-2 Model to Generate Protein Connect Map and Representations

The first time you run it, the model `esm2_t33_650M_UR50D` will be automatically downloaded. This will take a long time, which is normal.

```bash
python ./ESM-2/esm2_t33_650M_UR50D.py <protein_seq_fasta_file>  # <protein_seq_fasta_file> is the protein sequence file path
```
The model uses CUDA0 as the default training device. If you want to change it, use the following command:

```bash
python ./ESM-2/esm2_t33_650M_UR50D.py <protein_seq_fasta_file> [device_index]
```

For example, to generate the connect map file and representations file for the ATH948 dataset using CUDA1:

```bash
python ./ESM-2/esm2_t33_650M_UR50D.py datasets/protein_seq/ATH948_protein_seq.fa 1
```

This script uses default `window_size` and `stride` values of 1000 and 500 for processing long protein sequences. These are optimal parameters, but if you wish to adjust them, you can specify the parameters when running the code:

```bash
python ./ESM-2/esm2_t33_650M_UR50D.py <protein_seq_fasta_file> [device_index] [window_size] [stride]
```

#### 4.4.2 Use Connect Map and Representations Files to Generate Protein Graph Features

Run the following command to generate protein graph features:

```bash
python ./create_feature/protein_graph-feature.py <esm2_file_dir>  # <esm2_file_dir> is the directory where the connect map and representations files for the protein dataset are stored
```

For example, to generate protein graph features for the ATH948 dataset:

```bash
python ./create_feature/protein_graph-feature.py ESM-2/esm2-ATH948_protein_seq_w1000_s500
```

The output feature files will be stored by default in `./features/protein_graph-feature`. To store them in a different location, use the following command to specify the output directory:

```bash
python ./create_feature/protein_graph-feature.py <esm2_file_dir> <output_dir>  # <output_dir> is the directory where you want to store the feature files
```

## 5. Running SeqMG-RPI

After all the features of the required dataset have been generated, you can run the main program with the following command:

```bash
# The same dataset is used as training and test sets.
python main.py <dataset_pair_file>
# OR
# Different datasets are used as training and test sets
python main2.py <train_dataset_pair_file> <test_dataset_pair_file>
```

The model uses CUDA0 as the default training device. If you want to change it, use the following command:

```bash
python main.py <dataset_pair_file> [device_index]
# OR
python main2.py <train_dataset_pair_file> <test_dataset_pair_file> [device_index]
```

For example:

```bash
# Use the ATH948 dataset as training and test sets and run on device CUDA0
python main.py datasets/pair/ATH948_pairs.csv 0
# OR
# Use RPI1807 as training set, RPI_D as test set, and run on device CUDA1
python main2.py datasets/pair/RPI1807_pairs.csv datasets/multi-species/RPI_D/RPI_D_interaction.csv 1
```

If you have modified the feature file storage paths in the previous steps, please adjust the following lines in `main.py` and `main2.py`:

```bash
    rna_mutil_channel_feature_dir = "./features/RNA_multi-channel-features"
    rna_kmer_frequency_feature_dir = "./features/RNA_kmer-frequency-features"
    rna_sparse_matrix_feature_dir = "./features/RNA_sparse-matrix-features"
    protein_graph_feature_dir = "./features/protein_graph-feature"
```

Change them to the paths you have set:

```bash
    rna_mutil_channel_feature_dir = "<output_dir>"
    rna_kmer_frequency_feature_dir = "<output_dir>"
    rna_sparse_matrix_feature_dir = "<output_dir>"
    protein_graph_feature_dir = "<output_dir>"
```

The interaction probability files generated by `main.py` and `main2.py` will be stored in the `./result` directory.

## 6. Datasets

The datasets used in this study are stored in `./datasets/`

## Citation and contact

```bash
@article{
}
```
