"""
LipSyncD — Training Script
Configured for: Deepfakes + FaceSwap, c23 compression

Quick start:
  # 1. Verify your folder structure first:
  python train.py --data /path/to/FaceForensics++ --scan

  # 2. Run training (uses official train split if splits/ folder exists):
  python train.py --data /path/to/FaceForensics++

  # 3. Resume / use cached features (skips slow re-extraction):
  python train.py --data /path/to/FaceForensics++ --cache

  # 4. After training, restart the server:
  python app.py
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
from pathlib import Path

os.makedirs("weights", exist_ok=True)
os.makedirs("logs",    exist_ok=True)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    print("[ERROR] PyTorch not installed.\n  pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn missing — install for full metrics:\n  pip install scikit-learn")

from dataset import FFPPDataset, FeatureExtractor, FF_MANIPULATIONS, scan_dataset


# ─────────────────────────────────────────────────────────────
class DeepfakeClassifier(nn.Module):
    """
    4-layer MLP on top of EfficientNet-B0 CNN embeddings.
    Input: 2563-dim (2560 CNN mean+std + 3 sync features).
    """
    def __init__(self, input_dim: int = 2563, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.75),
            nn.Linear(256, 64),        nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
def extract_features(samples, extractor, desc="Extracting"):
    X, y, failed = [], [], 0
    total = len(samples)
    t0 = time.time()

    for i, (path, label) in enumerate(samples):
        if i % 10 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            eta  = (total - i - 1) / max(rate, 1e-6)
            bar  = "█" * int(20 * (i+1) / total) + "░" * (20 - int(20 * (i+1) / total))
            print(f"  [{desc}] |{bar}| {i+1}/{total}  {rate:.1f} v/s  ETA {eta:.0f}s  ", end='\r')

        feat = extractor.extract(path)
        if feat is not None:
            X.append(feat)
            y.append(label)
        else:
            failed += 1

    elapsed = time.time() - t0
    print(f"\n  [{desc}] ✓ {len(X)} ok  ✗ {failed} failed  ({elapsed:.1f}s total)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
def train(args):
    print("\n" + "="*60)
    print("  LipSyncD — Training on FaceForensics++")
    print("="*60)
    print(f"  Root         : {args.data}")
    print(f"  Compression  : {args.compression}")
    print(f"  Manipulations: {args.manip}")
    print(f"  Max/class    : {args.max_per_class}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Device       : {args.device}")
    print("="*60 + "\n")

    # ── 1. Scan dataset ───────────────────────────────────────
    print("[1/6] Loading dataset index...")
    ds = FFPPDataset(
        root=args.data,
        compression=args.compression,
        manipulations=args.manip,
        max_per_class=args.max_per_class,
        split="train" if not args.no_split else None,
    )
    _, val_samples = FFPPDataset(
        root=args.data,
        compression=args.compression,
        manipulations=args.manip,
        max_per_class=max(50, args.max_per_class // 5),
        split="val" if not args.no_split else None,
    ).train_val_split(val_ratio=1.0)  # use the val split entirely

    # If no official splits folder → fall back to random 85/15
    if not (Path(args.data) / "splits" / "train.json").exists() or args.no_split:
        print("  (No splits/ folder found — using random 85/15 split)")
        train_samples, val_samples = ds.train_val_split(val_ratio=0.15)
    else:
        train_samples = ds.samples

    print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}\n")

    # ── 2. Feature extraction (or load cache) ─────────────────
    print("[2/6] Extracting features...")
    cache_key  = f"cnn_features_{args.compression}_{'_'.join(sorted(args.manip))}"
    cache_path = Path("logs") / f"{cache_key}.npz"

    if args.cache and cache_path.exists():
        print(f"  Loading cache: {cache_path}")
        npz = np.load(cache_path)
        X_train, y_train = npz['X_train'], npz['y_train']
        X_val,   y_val   = npz['X_val'],   npz['y_val']
    else:
        extractor = FeatureExtractor(n_frames=20, device=args.device)
        X_train, y_train = extract_features(train_samples, extractor, "Train")
        X_val,   y_val   = extract_features(val_samples,   extractor, "Val  ")
        np.savez(cache_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        print(f"  Features cached → {cache_path}  (use --cache to reuse)")

    if len(X_train) == 0:
        print("\n[ERROR] Feature extraction returned 0 samples.")
        print("  → Run: python train.py --data /path/to/FF++ --scan")
        print("  → Check that your videos contain a visible face and audio.")
        sys.exit(1)

    print(f"\n  Feature dim  : {X_train.shape[1]}")
    print(f"  Train balance: real={int((y_train==0).sum())} fake={int((y_train==1).sum())}")
    print(f"  Val   balance: real={int((y_val==0).sum())}   fake={int((y_val==1).sum())}\n")

    # ── 3. Normalize ──────────────────────────────────────────
    print("[3/6] Fitting StandardScaler...")
    if SKLEARN_OK:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val   = scaler.transform(X_val).astype(np.float32)
        with open("weights/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("  Saved → weights/scaler.pkl\n")
    else:
        mean    = X_train.mean(axis=0);  std = X_train.std(axis=0) + 1e-8
        X_train = ((X_train - mean) / std).astype(np.float32)
        X_val   = ((X_val   - mean) / std).astype(np.float32)
        np.save("weights/scaler_mean.npy", mean)
        np.save("weights/scaler_std.npy",  std)
        print("  Saved → weights/scaler_mean/std.npy\n")

    # ── 4. Model ──────────────────────────────────────────────
    print("[4/6] Building model...")
    device    = torch.device(args.device)
    input_dim = X_train.shape[1]
    model     = DeepfakeClassifier(input_dim=input_dim, dropout=args.dropout).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  device: {device}\n")

    # Handle class imbalance
    n_real  = (y_train == 0).sum()
    n_fake  = (y_train == 1).sum()
    pw      = torch.tensor([n_real / max(n_fake, 1)], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False
    )

    # ── 5. Training loop ──────────────────────────────────────
    print("[5/6] Training...\n")
    best_auc, best_acc = 0.0, 0.0
    log_rows = []
    patience, patience_count = 12, 0   # early stopping

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(X_train)
        scheduler.step()

        # Validate
        model.eval()
        probs_all, labels_all, v_loss = [], [], 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += loss_fn(logits, yb).item() * len(xb)
                probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels_all.extend(yb.cpu().numpy().tolist())
        v_loss /= len(X_val)

        preds = (np.array(probs_all) > 0.5).astype(int)
        acc   = (accuracy_score(labels_all, preds) if SKLEARN_OK
                 else float(np.mean(np.array(labels_all) == preds)))
        auc   = (roc_auc_score(labels_all, probs_all)
                 if SKLEARN_OK and len(set(labels_all)) > 1 else 0.5)
        lr_now = optimizer.param_groups[0]['lr']

        marker = ""
        if auc > best_auc:
            best_auc, best_acc = auc, acc
            patience_count = 0
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'input_dim':      input_dim,
                'val_acc':        acc,
                'val_auc':        auc,
                'compression':    args.compression,
                'manipulations':  args.manip,
            }, "weights/model.pth")
            marker = "  ← best"
        else:
            patience_count += 1

        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"loss {t_loss:.4f}→{v_loss:.4f} | "
              f"acc {acc*100:.1f}% | AUC {auc:.4f} | lr {lr_now:.1e}{marker}")

        log_rows.append({'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
                         'acc': acc, 'auc': auc})

        if patience_count >= patience:
            print(f"\n  Early stopping triggered (no AUC improvement for {patience} epochs).")
            break

    # ── 6. Report ──────────────────────────────────────────────
    print(f"\n[6/6] Done.")
    print(f"  Best AUC : {best_auc:.4f}")
    print(f"  Best Acc : {best_acc*100:.1f}%")
    print(f"  Weights  : weights/model.pth\n")

    # Full classification report
    ckpt = torch.load("weights/model.pth", map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs_all.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
            labels_all.extend(yb.numpy().tolist())
    preds = (np.array(probs_all) > 0.5).astype(int)
    if SKLEARN_OK:
        print(classification_report(labels_all, preds, target_names=['Real', 'Fake']))

    with open("logs/training_log.json", "w") as f:
        json.dump({'log': log_rows, 'best_auc': best_auc, 'best_acc': best_acc}, f, indent=2)

    print("  Log saved → logs/training_log.json")
    print("\n  ✓ Restart the server to use the trained model:")
    print("    python app.py\n")


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LipSyncD Training — FaceForensics++ (Deepfakes + FaceSwap, c23)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --data /data/FF++\n"
            "  python train.py --data /data/FF++ --scan\n"
            "  python train.py --data /data/FF++ --cache\n"
            "  python train.py --data /data/FF++ --manip Deepfakes FaceSwap --compression c23\n"
        )
    )
    p.add_argument("--data",          required=True,
                   help="Path to the FaceForensics++ root folder")
    p.add_argument("--scan",          action="store_true",
                   help="Print folder structure and exit (no training)")
    p.add_argument("--compression",   default="c23",    choices=["c0", "c23", "c40"])
    p.add_argument("--manip",         default=None,     nargs="+",
                   choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                   help=f"Manipulation types (default: {FF_MANIPULATIONS})")
    p.add_argument("--max-per-class", type=int,  default=500,
                   help="Max videos per class — lower for faster test runs")
    p.add_argument("--epochs",        type=int,  default=60)
    p.add_argument("--batch-size",    type=int,  default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache",         action="store_true",
                   help="Load cached features from logs/ if available")
    p.add_argument("--no-split",      action="store_true",
                   help="Ignore splits/ folder, use random 85/15 split instead")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.manip is None:
        args.manip = FF_MANIPULATIONS    # defaults to ["Deepfakes", "FaceSwap"]

    # --scan: just print structure and exit
    if args.scan:
        scan_dataset(args.data)
        sys.exit(0)

    train(args)