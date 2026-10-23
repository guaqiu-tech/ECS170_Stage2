from local_code.stage_2_code.MNIST_CSV_Dataset import MNIST_CSV_Dataset
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Setting_MNIST_CSV import Setting_MNIST_CSV
from local_code.stage_2_code.Evaluate_Metrics import (
    Evaluate_Accuracy, Evaluate_F1_Macro, Evaluate_F1_Weighted,
    Evaluate_Recall_Macro, Evaluate_Precision_Macro)
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(12)
np.random.seed(12)

# Initialize dataset objects
train_ds = MNIST_CSV_Dataset('data/stage_2_data/train.csv', 'train')
test_ds = MNIST_CSV_Dataset('data/stage_2_data/test.csv', 'test')
setting = Setting_MNIST_CSV('MNIST_Setting')
setting.set_test_dataset(test_ds)

# E1 (Baseline):         USE_DEEP=False, USE_BN_DROPOUT=False
# E2 (Overfitting Test): USE_DEEP=True,  USE_BN_DROPOUT=False
# E3 (Final Model):      USE_DEEP=True,  USE_BN_DROPOUT=True
USE_DEEP = True
USE_BN_DROPOUT = True

# Initialize the model with the chosen configuration
mlp = Method_MLP(use_deep=USE_DEEP, use_bn_dropout=USE_BN_DROPOUT)

# Prepare and run the setting
setting.prepare(train_ds, mlp, None, Evaluate_Accuracy())
print(f'\n************ Start (Deep={USE_DEEP}, BN/Drop={USE_BN_DROPOUT}) ************')
learned_result, acc = setting.load_run_save_evaluate()

# Print all evaluation metrics
print('\n=== Final Evaluation Results ===')
metrics = [
    Evaluate_Accuracy(), Evaluate_F1_Macro(), Evaluate_F1_Weighted(),
    Evaluate_Recall_Macro(), Evaluate_Precision_Macro()
]
for ev in metrics:
    ev.data = learned_result
    print(f'{ev.evaluate_name}: {ev.evaluate():.4f}')

# Plot the training convergence curve
img_name = f'loss_deep_{USE_DEEP}_bn_{USE_BN_DROPOUT}.png'
plt.figure(figsize=(8, 5))
plt.plot(learned_result['loss_history'], marker='o')
plt.title(f'Loss Curve (Deep={USE_DEEP}, BN/Dropout={USE_BN_DROPOUT})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(img_name)
print(f'\nPlot saved as: {img_name}')
print('************ Finish ************')