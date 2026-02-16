# ============================================================
# SAFE MODE UrbanSound8K Training Script (Single File)
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# SAFE MODE GLOBAL SETTINGS
# ============================================================

torch.set_num_threads(4)
torch.backends.cudnn.enabled = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================

ROOT_PATH = Path(r"D:\Anish\Research\UrbanSound8k")
AUDIO_PATH = ROOT_PATH / "audio"
METADATA_PATH = ROOT_PATH / "metadata" / "UrbanSound8K.csv"
CHECKPOINT_PATH = Path("./Checkpoint")
CHECKPOINT_PATH.mkdir(exist_ok=True)

# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 16
MAX_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 3e-4
PATIENCE = 20

SAMPLE_RATE = 22050
MAX_AUDIO_LENGTH_SEC = 4
MAX_AUDIO_LENGTH = SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
NUM_CLASSES = 10

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):
    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        audio = librosa.util.pad_center(audio, size=MAX_AUDIO_LENGTH)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    contrast = librosa.util.fix_length(
        contrast, size=logmel.shape[0], axis=0
    )

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    features = np.stack(
        [norm(logmel), norm(delta), norm(delta2), norm(contrast)],
        axis=0
    ).astype(np.float32)

    return torch.tensor(features)

# ============================================================
# WAVE AUGMENTATION
# ============================================================

wave_aug = Compose([
    AddGaussianNoise(0.001, 0.01, p=0.4),
    TimeStretch(0.9, 1.1, p=0.3),
    PitchShift(-3, 3, p=0.3),
])

# ============================================================
# DATASET
# ============================================================

class UrbanSoundDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = AUDIO_PATH / f"fold{row.fold}" / row.slice_file_name

        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, size=MAX_AUDIO_LENGTH)

        if self.train:
            y = wave_aug(samples=y, sample_rate=SAMPLE_RATE)

        x = extract_features(y)
        return x, int(row.classID)

# ============================================================
# MODEL
# ============================================================

class ChannelGate(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DWRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(dim * 1.5)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.act(self.pw(x))
        x = self.proj(x)
        x = self.bn(x)
        return r + x

class ParallelBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.global_branch = nn.Conv2d(
            dim, dim, 5, padding=4, dilation=2, groups=dim, bias=False
        )
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.act = nn.GELU()
        self.gate = ChannelGate(dim)

    def forward(self, x):
        l = self.local(x)
        g = self.global_branch(x)
        x = torch.cat([l, g], dim=1)
        x = self.act(self.fuse(x))
        return self.gate(x)

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.stage1 = nn.Sequential(ParallelBlock(64), DWRes(64))
        self.down1 = nn.Conv2d(64, 128, 2, stride=2, bias=False)

        self.stage2 = nn.Sequential(
            ParallelBlock(128),
            DWRes(128),
            DWRes(128)
        )
        self.down2 = nn.Conv2d(128, 240, 2, stride=2, bias=False)

        self.stage3 = nn.Sequential(
            ParallelBlock(240),
            DWRes(240),
            DWRes(240),
            DWRes(240)
        )

        self.norm = nn.BatchNorm2d(240)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(240),
            nn.Dropout(0.3),
            nn.Linear(240, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.norm(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(model, loader, optimizer=None, desc="Train"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0
    preds, labels = [], []

    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.set_grad_enabled(train):
            out = model(x)
            loss = loss_fn(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        preds.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, total_loss / total

# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(METADATA_PATH)

    train_df = df[df.fold.isin(TRAIN_FOLDS)]
    val_df = df[df.fold == VAL_FOLD]
    test_df = df[df.fold == TEST_FOLD]

    train_dataset = UrbanSoundDataset(train_df, True)
    val_dataset = UrbanSoundDataset(val_df, False)
    test_dataset = UrbanSoundDataset(test_df, False)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=0)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, False, num_workers=0)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers=0)

    model = Model().to(DEVICE)
    global loss_fn
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, patience = 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_acc, tr_f1, _ = run_epoch(model, train_loader, optimizer, "Train")
        va_acc, va_f1, _ = run_epoch(model, val_loader, None, "Valid")

        print(f"Epoch {epoch:03d} | TR F1 {tr_f1:.4f} | VA F1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            patience = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH / "best.pth")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(torch.load(CHECKPOINT_PATH / "best.pth"))
    model.eval()

    _, te_f1, _ = run_epoch(model, test_loader, None, "Test")
    print(f"TEST F1: {te_f1:.4f}")

if __name__ == "__main__":
    main()
