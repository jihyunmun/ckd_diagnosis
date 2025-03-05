import torch
import torch.nn as nn
import torch.optim as optim
from utils import compute_metrics
from tqdm import tqdm

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds, all_labels = [], []

    pbar = tqdm(data_loader, desc='Training')
    for _, batch in enumerate(pbar):
        spec_data, glottal_features, labels = batch
        spec_data = spec_data.to(device)
        glottal_features = glottal_features.to(device)
        label_tensor = labels['diag'].to(device)

        optimizer.zero_grad()

        y_pred, _ = model(spec_data, glottal_features)

        y_pred = y_pred.squeeze(dim=1)
        loss = criterion(y_pred, label_tensor)

        loss.backward()
        optimizer.step()

        batch_size = y_pred.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(y_pred.detach().cpu())
        all_labels.append(label_tensor.detach().cpu())

        pbar.set_postfix({'loss': total_loss / total_samples})

    average_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy, f1, precision, recall = compute_metrics(all_preds, all_labels)
    return average_loss, (accuracy, f1, precision, recall)

def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation')
        for _, batch in enumerate(pbar):
            spec_data, glottal_features, labels = batch
            spec_data = spec_data.to(device)
            glottal_features = glottal_features.to(device)
            label_tensor = labels['diag'].to(device)

            y_pred, _ = model(spec_data, glottal_features)

            y_pred = y_pred.squeeze(dim=1)
            loss = criterion(y_pred, label_tensor)

            batch_size = y_pred.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(y_pred.detach().cpu())
            all_labels.append(label_tensor.detach().cpu())

            pbar.set_postfix({'loss': total_loss / total_samples})

    average_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy, f1, precision, recall = compute_metrics(all_preds, all_labels)
    return average_loss, (accuracy, f1, precision, recall)