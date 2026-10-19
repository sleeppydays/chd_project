from tqdm import tqdm
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold

from pipeline_utils import (
    ARCH_LIST, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    get_model, compute_binary_metrics, get_group_id,
)

# ==========================================
# CONFIG — edit these
# ==========================================
DATA_ROOT = "/home/aaush/cardium_native/processed_C_adaptive"
MODEL_OUT_DIR = "/home/aaush/cardium_native/models"
REPORT_DIR = "/home/aaush/cardium_native"
POSITIVE_CLASS = "chd"
N_FOLDS = 5
EPOCHS_PER_FOLD = 8
BATCH_SIZE = 32
LR = 1e-4
SEED = 42

# Flip this on for a fast sanity pass before committing to the full
# 3-architecture x 5-fold run (15 separate training runs -- budget real
# time for the full thing even on lightweight models).
QUICK_SANITY_RUN = False
if QUICK_SANITY_RUN:
    N_FOLDS, EPOCHS_PER_FOLD = 2, 2

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print(f"Device: {device}")


# No flips: flipping is not a valid augmentation for anatomical/scan
# orientation (see chat).
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CardiumDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (filepath, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ==========================================
# GATHER SAMPLES + GROUPS
# ==========================================
class_names = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
if len(class_names) != 2:
    raise ValueError(f"Expected exactly 2 classes under {DATA_ROOT}, found {class_names}.")
if POSITIVE_CLASS not in class_names:
    raise ValueError(f"POSITIVE_CLASS='{POSITIVE_CLASS}' not among {class_names}.")
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(f"Classes: {class_names}")

samples, groups = [], []
for c in class_names:
    class_dir = os.path.join(DATA_ROOT, c)
    for fn in sorted(os.listdir(class_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            fp = os.path.join(class_dir, fn)
            samples.append((fp, class_to_idx[c]))
            groups.append(get_group_id(fn))

labels = np.array([lbl for _, lbl in samples])
groups = np.array(groups)
n_unique_groups = len(set(groups))
print(f"Total samples: {len(samples)} | unique groups: {n_unique_groups}")
if n_unique_groups == len(samples):
    print(
        "WARNING: every image resolved to its own group (get_group_id() was not "
        "customized). If multiple images per patient/study exist in your dataset, "
        "this CV does NOT protect against patient-level leakage -- edit "
        "get_group_id() above before trusting these numbers for a paper."
    )

# ==========================================
# TRAIN / EVAL ONE FOLD
# ==========================================
def train_one_fold(arch, train_samples, val_samples):
    # ADDED pin_memory=True to speed up CPU-to-GPU transfer!
    train_loader = DataLoader(CardiumDataset(train_samples, train_tf),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(CardiumDataset(val_samples, val_tf),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = get_model(arch, len(class_names)).to(device)
    train_labels = [lbl for _, lbl in train_samples]
    counts = np.bincount(train_labels, minlength=len(class_names))
    weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
    weights = (weights / weights.sum() * len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        # WRAPPED IN tqdm FOR A LIVE PROGRESS BAR!
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PER_FOLD} [{arch}]"):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    model.eval()
    pos_idx = class_names.index(POSITIVE_CLASS)
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            outputs = model(imgs.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_scores.extend(probs[:, pos_idx].cpu().numpy())

    metrics = compute_binary_metrics(all_labels, all_preds, all_scores, class_names, POSITIVE_CLASS)
    return model, metrics


# ==========================================
# CROSS-VALIDATE EACH ARCHITECTURE
# ==========================================
print(f"\nRunning {len(ARCH_LIST)} architectures x {N_FOLDS} folds x {EPOCHS_PER_FOLD} epochs "
      f"= {len(ARCH_LIST) * N_FOLDS} training runs. This takes a while -- consider "
      f"QUICK_SANITY_RUN=True for a first pass.\n")

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_fold_results = []

for arch in ARCH_LIST:
    print(f"=== {arch} ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(samples)), labels, groups), 1):
        train_samples = [samples[i] for i in tr_idx]
        val_samples = [samples[i] for i in va_idx]
        _, metrics = train_one_fold(arch, train_samples, val_samples)
        metrics.update({"architecture": arch, "fold": fold})
        all_fold_results.append(metrics)
        printable = {k: round(v, 3) for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"  fold {fold}: {printable}")

results_df = pd.DataFrame(all_fold_results)
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "sensitivity", "specificity"]
summary = results_df.groupby("architecture")[metric_cols].agg(["mean", "std"])
summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
summary = summary.reset_index()

print("\n=== Cross-validation summary (mean +/- std across folds) ===")
print(summary.to_string(index=False))

md_path = os.path.join(REPORT_DIR, "cv_benchmark_summary.md")
with open(md_path, "w") as f:
    f.write("# Cross-Validation Benchmark Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write(f"\n\n{N_FOLDS}-fold StratifiedGroupKFold, {EPOCHS_PER_FOLD} epochs/fold. ")
    f.write("Grouping status: " + (
        "no real patient/study grouping detected -- see script warning.\n"
        if n_unique_groups == len(samples) else "grouped by get_group_id().\n"
    ))

log_path = os.path.join(REPORT_DIR, "pipeline_log.xlsx")
with pd.ExcelWriter(log_path, engine="openpyxl",
                     mode="a" if os.path.exists(log_path) else "w",
                     if_sheet_exists="replace" if os.path.exists(log_path) else None) as writer:
    results_df.to_excel(writer, sheet_name="cv_fold_results", index=False)
    summary.to_excel(writer, sheet_name="cv_summary", index=False)

print(f"\nWritten: {md_path}")
print("Written: cv_fold_results / cv_summary sheets in pipeline_log.xlsx")

# ==========================================
# SELECT BEST ARCHITECTURE, REFIT ON FULL DATA
# ==========================================
best_arch = summary.loc[summary["f1_mean"].idxmax(), "architecture"]
print(f"\nBest architecture by mean F1 across folds: {best_arch}")
print("Refitting on the full dataset for the deployable checkpoint used by script 3...")

full_loader = DataLoader(
    CardiumDataset(samples, train_tf),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
final_model = get_model(best_arch, len(class_names)).to(device)
counts = np.bincount(labels, minlength=len(class_names))
weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
weights = (weights / weights.sum() * len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)

for epoch in range(EPOCHS_PER_FOLD):
    final_model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(full_loader, desc=f"Refit {epoch+1}/{EPOCHS_PER_FOLD} [{best_arch}]"):
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(final_model(imgs), lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    print(f"  refit epoch {epoch+1}/{EPOCHS_PER_FOLD} loss={running_loss/len(samples):.4f}")
