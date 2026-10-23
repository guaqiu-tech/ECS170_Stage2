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
