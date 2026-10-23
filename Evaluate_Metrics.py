from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np

# Helper function to convert PyTorch tensors to numpy arrays
def _to_numpy(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

class Evaluate_Accuracy(evaluate):
    def __init__(self): super().__init__('Accuracy', 'Accuracy')
    def evaluate(self):
        # Calculate standard accuracy
        return accuracy_score(_to_numpy(self.data['true_y']), _to_numpy(self.data['pred_y']))

class Evaluate_F1_Macro(evaluate):
    def __init__(self): super().__init__('F1_macro', 'Macro F1')
    def evaluate(self):
        # Calculate macro-averaged F1 score (treats all classes equally)
        return f1_score(_to_numpy(self.data['true_y']), _to_numpy(self.data['pred_y']), average='macro')

class Evaluate_F1_Weighted(evaluate):
    def __init__(self): super().__init__('F1_weighted', 'Weighted F1')
    def evaluate(self):
        # Calculate weighted F1 score (accounts for class imbalance)
        return f1_score(_to_numpy(self.data['true_y']), _to_numpy(self.data['pred_y']), average='weighted')

class Evaluate_Recall_Macro(evaluate):
    def __init__(self): super().__init__('Recall_macro', 'Macro Recall')
    def evaluate(self):
        return recall_score(_to_numpy(self.data['true_y']), _to_numpy(self.data['pred_y']), average='macro')

class Evaluate_Precision_Macro(evaluate):
    def __init__(self): super().__init__('Precision_macro', 'Macro Precision')
    def evaluate(self):
        return precision_score(_to_numpy(self.data['true_y']), _to_numpy(self.data['pred_y']), average='macro')