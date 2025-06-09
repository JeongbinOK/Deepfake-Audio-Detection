import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from data_preprocess import load_data, load_config

from model.best_ecapa import build_model
from model.loss import FocalLoss
from torch_ema import ExponentialMovingAverage
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_model_summary

import random

def set_seed(seed=2025):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()  

def save_loss_plot(train_hist, val_hist, path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_hist, label="train")
    if val_hist: plt.plot(val_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(path); plt.close()

def lr_lambda(total_warmup):
    def fn(step): return (step + 1) / total_warmup if step < total_warmup else 1.0
    return fn

def main():
    cfg = load_config()
    # --------- Device ---------
    device = torch.device('cuda' if cfg.get('cuda', False) and torch.cuda.is_available()
                          else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print("device:", device)

    # ---------- Dataset ------------
    batch_size = cfg.get("batch_size")
    
    # cfg['use_specaugment']=False
    X_train, y_train, _ = load_data(
        cfg['train_metadata_path'], cfg['train_data_path'],
        cfg.get('sr', 16000), cfg.get('n_mfcc', 20), cfg
    )
    X_train = torch.from_numpy(np.array(X_train, dtype=np.float32)).unsqueeze(1)
    y_train = torch.from_numpy(np.array(y_train, dtype=np.int64)).long()

    total_samples = len(y_train)
    val_size = int(0.2 * total_samples)
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    labels = y_shuffled.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_idx, val_idx in sss.split(np.zeros(len(labels)), labels):
        X_train_split = X_shuffled[train_idx]
        y_train_split = y_shuffled[train_idx]
        X_val_split = X_shuffled[val_idx]
        y_val_split = y_shuffled[val_idx]
    unique, counts = np.unique(y_val_split.numpy(), return_counts=True)
    print(">>> [Validation set class counts]")      
    print(f"   Fake(0): {counts[unique.tolist().index(0)]}  |  Real(1): {counts[unique.tolist().index(1)]}")
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val_split, y_val_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------ Model ------
    dropout_p = cfg.get('dropout_p')
    depth = cfg.get('depth')
    n_mels = X_train.size(2)
    print("n_mels:", n_mels)
    model = build_model(
        input_dim=n_mels,
        hidden1=cfg.get('ecapa_channels', 512),
        hidden2=cfg.get('ecapa_emb_dim', 192),
        num_classes=cfg.get('num_classes', 2),
        dropout_p=dropout_p,
        depth=depth,
    ).to(device)
    
    dummy = torch.zeros(1, 1, n_mels, cfg.get('max_time_steps', 300)).to(device)
    print("\nModel Summary:")
    print(pytorch_model_summary.summary(
        model, dummy,
        max_depth=2,
        show_input=False,
    ))

    # ----- Loss / Optim -----
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    if cfg.get('use_focal_loss', False):
        criterion = FocalLoss(alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    lr         = float(cfg.get('lr', 1e-3))
    weight_dec = float(cfg.get('weight_decay', 0.0))
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_dec)
    print(f"[OPTIM] AdamW lr={lr}  weight_decay={weight_dec}")

    # ------ Scheduler ------
    sched_cfg = cfg.get('lr_scheduler', {})
    eta_min = float(sched_cfg.get('eta_min', 0.0))

    steps_per_epoch = len(train_loader)
    total_steps     = cfg['epochs'] * steps_per_epoch
    warmup          = int(cfg.get('warmup_steps', 500))

    sched_cfg = cfg.get('lr_scheduler', {})
    eta_min = float(sched_cfg.get('eta_min', 0.0))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min
    )


    # --------- Train ---------
    best_acc = 0.0
    tr_hist, val_hist = [], []
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            emb, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step() 
            ema.update(model.parameters())
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        tr_hist.append(running_loss / total)
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                emb, logits = model(xb)
                v_l = criterion(logits, yb)
                v_loss += v_l.item() * xb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += xb.size(0)
        val_hist.append(v_loss / v_total)
        val_acc = v_correct / v_total


        print(f"Epoch {epoch}/{cfg['epochs']} - Loss: {tr_hist[-1]:.4f} | "
              f"Val Loss: {val_hist[-1]:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg['model_path'])
            print("  Saved best model")
        

    save_loss_plot(tr_hist, val_hist, cfg.get('loss_plot_path', 'loss_plot.png'))
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

    
if __name__ == '__main__':
    main()