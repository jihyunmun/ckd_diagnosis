import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from config import Config
from dataset import SpectrogramGlottalDataset
from model import SpecGlottalTransformerModel
from utils import compute_pos_weight


###############################################################
# (1) Attention Weight
###############################################################
def ensemble_attention_weights(models, spec_data, glottal_data):
    """
    models: list of model
    spec_data: (B, 3, H, W)
    glottal_data: (B, 9)
    Return: average attention weights shape (B, num_heads, 2, 2)
    """
    with torch.no_grad():
        attn_sum = None
        for m in models:
            m.eval()
            _, attn_w = m(spec_data, glottal_data)
            if attn_sum is None:
                attn_sum = attn_w
            else:
                attn_sum += attn_w
        avg_attn = attn_sum / len(models)
    return avg_attn  # (B, num_heads, 2, 2)

###############################################################
# (2) Saliency(gradient) for glottal features
###############################################################
def compute_ensemble_glottal_saliency(models, spec_data, glottal_data, diag_labels, pos_weight, device):
    for m in models:
        m.eval()

    # spec_data는 gradient를 추적 안 함
    spec_data = spec_data.to(device)
    spec_data.requires_grad = False

    glottal_data = glottal_data.clone().detach().to(device).requires_grad_(True)
    diag_labels = diag_labels.float().to(device)

    total_logits = 0
    for m in models:
        logit, _ = m(spec_data, glottal_data)
        total_logits += logit
    ensemble_logits = total_logits / len(models)
    ensemble_logits = ensemble_logits.squeeze(1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(ensemble_logits, diag_labels)

    for m in models:
        m.zero_grad()
    if glottal_data.grad is not None:
        glottal_data.grad.zero_()

    loss.backward()

    grad_glottal = glottal_data.grad.detach()  # (B, 9)

    return grad_glottal

###############################################################
# (3) Labelwise Average Attention & Glottal Saliency
###############################################################
def compute_labelwise_attention_average(models, data_loader, device):
    """
    Return: avg_attn_0 (num_heads, 2, 2), count_0,
            avg_attn_1 (num_heads, 2, 2), count_1
    """
    sum_attn_0 = None
    count_0 = 0
    sum_attn_1 = None
    count_1 = 0

    for batch in data_loader:
        spec_data, glottal_data, labels = batch
        diag_labels = labels['diag']
        B = spec_data.size(0)

        with torch.no_grad():
            attn_w = ensemble_attention_weights(models, spec_data.to(device), glottal_data.to(device))
            # (B, num_heads, 2, 2)

        diag_labels_np = diag_labels.cpu().numpy()

        idx_0 = np.where(diag_labels_np == 0)[0]
        idx_1 = np.where(diag_labels_np == 1)[0]
        attn_w_cpu = attn_w.cpu()

        if len(idx_0) > 0:
            attn_0_batch = attn_w_cpu[idx_0].sum(dim=0) # (num_heads, 2, 2)
            if sum_attn_0 is None:
                sum_attn_0 = attn_0_batch
            else:
                sum_attn_0 += attn_0_batch
            count_0 += len(idx_0)

        if len(idx_1) > 0:
            attn_1_batch = attn_w_cpu[idx_1].sum(dim=0)
            if sum_attn_1 is None:
                sum_attn_1 = attn_1_batch
            else:
                sum_attn_1 += attn_1_batch
            count_1 += len(idx_1)

    avg_attn_0 = (sum_attn_0 / count_0).cpu().numpy() if count_0 > 0 else None
    avg_attn_1 = (sum_attn_1 / count_1).cpu().numpy() if count_1 > 0 else None

    return avg_attn_0, count_0, avg_attn_1, count_1


def compute_labelwise_glottal_saliency_average(models, data_loader, pos_weight, device):
    """
    Return: avg_glottal_0 (9, ), count_0,
            avg_glottal_1 (9, ), count_1
    """
    sum_glottal_0 = None
    count_0 = 0
    sum_glottal_1 = None
    count_1 = 0

    for batch in data_loader:
        spec_data, glottal_data, labels = batch
        diag_labels = labels['diag']
        B = spec_data.size(0)

        grad_glottal = compute_ensemble_glottal_saliency(
            models, spec_data, glottal_data, diag_labels, pos_weight, device
        )
        grad_glottal_abs = grad_glottal.abs().cpu()  # (B, 9)
        diag_labels_np = diag_labels.cpu().numpy()

        idx_0 = np.where(diag_labels_np == 0)[0]
        idx_1 = np.where(diag_labels_np == 1)[0]

        if len(idx_0) > 0:
            glot_0 = grad_glottal_abs[idx_0].sum(dim=0)  # (9,)
            if sum_glottal_0 is None:
                sum_glottal_0 = glot_0
            else:
                sum_glottal_0 += glot_0
            count_0 += len(idx_0)

        if len(idx_1) > 0:
            glot_1 = grad_glottal_abs[idx_1].sum(dim=0)  # (9,)
            if sum_glottal_1 is None:
                sum_glottal_1 = glot_1
            else:
                sum_glottal_1 += glot_1
            count_1 += len(idx_1)

    avg_glottal_0 = (sum_glottal_0 / count_0).numpy() if count_0 > 0 else None
    avg_glottal_1 = (sum_glottal_1 / count_1).numpy() if count_1 > 0 else None

    return avg_glottal_0, count_0, avg_glottal_1, count_1

###############################################################
# (4) Visualization: 2×2 Attention Heatmap + Glottal Bar Plot
###############################################################
def visualize_attention_heatmap(avg_attn, label_name="Non-CKD"):
    """
    avg_attn: shape (num_heads, 2, 2) -> (2,2)
    """
    if avg_attn is None:
        print(f"No data for {label_name}")
        return
    # average multiple heads
    mat = avg_attn.mean(axis=0)  # (2,2)
    tokens = ["Spectrogram", "Glottal"]
    plt.figure(figsize=(3,3))
    sns.heatmap(mat, annot=True, cmap='Blues', xticklabels=tokens, yticklabels=tokens, vmin=0, vmax=1)
    plt.title(f"{label_name} Average Attention")
    plt.show()

def visualize_glottal_barplot(avg_glottal, label_name="Non-CKD"):
    """
    avg_glottal: shape (9,)
    """
    if avg_glottal is None:
        print(f"No glottal data for {label_name}")
        return

    feature_names = [
        'GCI var','NAQ avg','NAQ std','QOQ avg','QOQ std',
        'H1H2 avg','H1H2 std','HRF avg','HRF std'
    ]
    plt.figure(figsize=(5,4))
    plt.bar(range(len(feature_names)), avg_glottal)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.title(f"{label_name} Average Glottal Feature Saliency")
    plt.tight_layout()
    plt.show()

###############################################################
# (5) Main
###############################################################
def run_analysis():
    print("[INFO] Analysis mode: Using test set for analysis")

    test_dataset = SpectrogramGlottalDataset(
        spec_dir=Config.SPEC_TEST_DIR,
        glottal_csv=Config.TEST_GLOTTAL_CSV,
        normalize_glottal=False
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)

    models = []
    used_seed_count = 0
    for seed in Config.SEEDS:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_seed_{seed}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue
        model = SpecGlottalTransformerModel()
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model.eval()
        models.append(model)
        used_seed_count += 1

    if used_seed_count == 0:
        print("[ERROR] No valid checkpoints loaded.")
        return
    
    pos_weight_val = compute_pos_weight(test_loader)
    pos_weight_val = torch.tensor(pos_weight_val).to(Config.DEVICE)

    avg_attn_0, count_0, avg_attn_1, count_1 = compute_labelwise_attention_average(models, test_loader, Config.DEVICE)
    avg_glottal_0, count_0, avg_glottal_1, count_1 = compute_labelwise_glottal_saliency_average(models, test_loader, pos_weight_val, Config.DEVICE)

    print("[INFO] Visualizing Attention Heatmaps ...")
    visualize_attention_heatmap(avg_attn_0, label_name="Non-CKD")
    visualize_attention_heatmap(avg_attn_1, label_name="CKD")

    print("[INFO] Visualizing Glottal Feature Saliency ...")
    visualize_glottal_barplot(avg_glottal_0, label_name="Non-CKD")
    visualize_glottal_barplot(avg_glottal_1, label_name="CKD")

    print("[INFO] Analysis complete.")