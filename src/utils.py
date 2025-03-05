import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(preds, labels):
    y_pred_sigmoid = torch.sigmoid(preds).cpu().detach().numpy()
    y_pred = (y_pred_sigmoid > 0.5).astype(int)

    accuracy = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='macro')
    precision = precision_score(labels, y_pred, average='macro')
    recall = recall_score(labels, y_pred, average='macro')

    return accuracy, f1, precision, recall

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

def compute_pos_weight(data_loader):
    """
    pos_weight = (N - n_pos) / n_pos
    """

    pos_count = 0
    total_count = 0

    for batch in data_loader:
        _, labels = batch
        y_true = labels['diag']
        pos_count += (y_true == 1).sum().item()
        total_count += len(y_true)

    neg_count = total_count - pos_count

    if pos_count == 0:
        pos_weight = 1.0
    else:
        pos_weight = neg_count / float(pos_count)
    return pos_weight

def save_inference_results(y_true, y_pred, accuracy, f1, precision, recall, output_path):
    y_pred_list = y_pred.tolist()
    y_true_list = y_true.tolist()

    results = {
        "test_metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        },
        "y_true": y_true_list,
        "y_pred": y_pred_list
    }

    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Test results saved to {output_path}")

