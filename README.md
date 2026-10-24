# ECS 170 Stage 2: MLP for MNIST Classification

## 1. Project Structure
This repository contains the implementation of a Multi-Layer Perceptron (MLP) for the MNIST dataset using PyTorch, following the ECS 170 template structure.

Files included:
- `MNIST_CSV_Dataset.py`: Data loader for train/test CSV files.
- `Method_MLP.py`: MLP model architecture and training/testing loops.
- `Evaluate_Metrics.py`: Multi-class evaluation metrics (Accuracy, F1, Precision, Recall).
- `Setting_MNIST_CSV.py`: Experiment orchestrator connecting data, method, and evaluation.
- `main.py`: The main script to execute the experiments (E1-E5).

## 2. How to Run
To reproduce the results in the report, please place these 5 files in the correct directory path:
`local_code/stage_2_code/`

Ensure the dataset is located at:
`data/stage_2_data/train.csv` and `data/stage_2_data/test.csv`

Then, run the main script from the project root directory:
```bash
python -m local_code.stage_2_code.main

## 3. Ablation Study Reproduction Guide (E1 - E5)
To reproduce the 5 experiments mentioned in the report, please follow the settings below. All experiments use a fixed random seed (12) in `main.py`.

### Code Switching Instructions:
- **Model Depth & Regularization**: Modify `USE_DEEP` and `USE_BN_DROPOUT` in `main.py`.
- **Optimizer (E4)**: In `Method_MLP.py`, replace `torch.optim.Adam` with `torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)`.
- **Loss Function (E5)**: In `Method_MLP.py`, replace `nn.CrossEntropyLoss()` with `nn.CrossEntropyLoss(label_smoothing=0.1)`.

### Experiment Configurations:
- **E1 (Baseline)**: 
  - `USE_DEEP = False`, `USE_BN_DROPOUT = False` (in main.py)
  - Standard Adam & CrossEntropyLoss
- **E2 (Overfitting Test)**:
  - `USE_DEEP = True`, `USE_BN_DROPOUT = False` (in main.py)
  - Standard Adam & CrossEntropyLoss
- **E3 (Regularization)**:
  - `USE_DEEP = True`, `USE_BN_DROPOUT = True` (in main.py)
  - Standard Adam & CrossEntropyLoss
- **E4 (Different Optimizer)**:
  - `USE_DEEP = True`, `USE_BN_DROPOUT = True` (in main.py)
  - **SGD Optimizer** (lr=0.01, momentum=0.9) & Standard CrossEntropyLoss
- **E5 (Different Loss Function - Best Model)**:
  - `USE_DEEP = True`, `USE_BN_DROPOUT = True` (in main.py)
  - Standard Adam & **Label Smoothing CrossEntropyLoss** (0.1)
