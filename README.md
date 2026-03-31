# ST-HyperGCL: Spatio-temporal Hypergraph Contrastive Learning for Corporate Violation Risk Detection

This repository contains the official PyTorch implementation for the paper: **"ST-HyperGCL: Spatio-temporal Hypergraph Contrastive Learning Network"**.

## 📌 Directory Structure

- `data/`: Contains the processed tensors (`.pt` files). *Note: Due to CSMAR database commercial licensing, we provide a synthetic Toy Dataset here for reproducibility.*
- `models/`: Contains the core neural network architectures.
  - `st_hypergcl.py`: The proposed ST-HyperGCL model (Residual GNN + GRU + GCL).
  - `baseline.py`: The MLP baseline model using pure financial features.
- `scripts/`: Data preprocessing scripts used to construct dynamic hypergraphs from raw CSV files.
- `notebooks/`: Exploratory Data Analysis (EDA) notebooks.
- `train_ultimate.py`: Entry script to train and evaluate the ST-HyperGCL model.
- `train_baseline.py`: Entry script to train and evaluate the MLP baseline.

## 🚀 How to Run

**1. Environment Setup**
Ensure you have Python 3.8+ and PyTorch installed. 
```bash
pip install torch pandas scikit-learn numpy
2. Run the Baseline (MLP)
To reproduce the baseline performance (pure financial tabular data without topological structures):

Bash
python train_baseline.py

3. Run the ST-HyperGCL (Ours)
To reproduce the final results utilizing spatio-temporal hypergraphs and contrastive learning:

Bash
python train_ultimate.py
Expected Output: The model should converge around 200 epochs and reach an AUC of ~0.793 on the evaluation set.

⚖️ Data Privacy & Copyright
The raw corporate financial statements and R&D alliance data are sourced from the CSMAR and CNRDS databases. According to their strict commercial data protection agreements, we cannot publicly distribute the raw .csv files. Researchers can apply for raw data access directly from their official platforms. We provide a toy .pt dataset strictly for code execution verification.