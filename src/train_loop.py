import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config, set_seed
from utils import EarlyStopping, compute_pos_weight, compute_metrics, save_inference_results
from dataset import SpectrogramGlottalDataset
from model import SpecGlottalTransformerModel
from train import initialize_weights, train_one_epoch, validate_one_epoch


def run_train():
    print("[INFO] Starting training...")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 예: Spectrogram + Glottal Dataset 사용
    train_dataset = SpectrogramGlottalDataset(
        spec_dir=Config.SPEC_TRAIN_DIR,
        glottal_csv=Config.TRAIN_GLOTTAL_CSV,
        normalize_glottal=False
    )
    valid_dataset = SpectrogramGlottalDataset(
        spec_dir=Config.SPEC_VALID_DIR,
        glottal_csv=Config.VALID_GLOTTAL_CSV,
        normalize_glottal=False
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=Config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)

    for seed in Config.SEEDS:
        set_seed(seed)
        print(f"[Seed={seed}]")

        model = SpecGlottalTransformerModel(
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
            model_name='resnet18',
            pretrained=True
        ).to(Config.DEVICE)
        model.apply(initialize_weights)

        if seed == Config.SEEDS[0]:
            pos_weight = compute_pos_weight(train_loader)
            pos_weight = torch.tensor(pos_weight).to(Config.DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=0)

        best_loss = float('inf')

        for epoch in range(Config.NUM_EPOCHS):
            print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")
            train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
            valid_loss, valid_metrics = validate_one_epoch(model, valid_loader, criterion, Config.DEVICE)

            print(f"Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}")
            print(f"Train Metrics={train_metrics}, Valid Metrics={valid_metrics}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_seed_{seed}.pth")
                torch.save(model.state_dict(), ckpt_path)

            scheduler.step(valid_loss)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("[INFO] Early stopping triggered.")
                break

    print("[INFO] Training complete.")

def run_inference():
    print("[INFO] Starting inference...")

    test_dataset = SpectrogramGlottalDataset(
        spec_dir=Config.SPEC_TEST_DIR,
        glottal_csv=Config.TEST_GLOTTAL_CSV,
        normalize_glottal=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=Config.NUM_WORKERS)

    y_pred_total = torch.zeros(len(test_dataset)).to(Config.DEVICE)
    y_true_total = torch.zeros(len(test_dataset)).to(Config.DEVICE)

    used_seed_count = 0

    start_idx = 0
    for seed in Config.SEEDS:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_seed_{seed}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue

        model = SpecGlottalTransformerModel(
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            dropout=0.1,
            model_name='resnet18',
            pretrained=True
        ).to(Config.DEVICE)

        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model.eval()
        used_seed_count += 1

        batch_start = 0
        with torch.no_grad():
            for batch in test_loader:
                spec_data, glottal_features, labels = batch
                spec_data = spec_data.to(Config.DEVICE)
                glottal_features = glottal_features.to(Config.DEVICE)
                diag_labels = labels['diag'].to(Config.DEVICE)

                y_pred, _ = model(spec_data, glottal_features)
                y_pred = y_pred.squeeze(1)

                bsz = y_pred.size(0)
                y_pred_total[batch_start:batch_start+bsz] += y_pred
                if seed == Config.SEEDS[0]:
                    y_true_total[batch_start:batch_start+bsz] = diag_labels
                batch_start += bsz

    if used_seed_count == 0:
        print("[ERROR] No valid checkpoints loaded.")
        return

    y_pred_total = y_pred_total / used_seed_count 
    accuracy, f1, precision, recall = compute_metrics(y_pred_total, y_true_total)
    print(f"Test metrics: Acc={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    save_inference_results(
        y_true_total.cpu(),
        y_pred_total.cpu(),
        accuracy,
        f1,
        precision,
        recall,
        Config.OUTPUT_JSON
    )