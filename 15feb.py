import os
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import librosa
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

torch.set_flush_denormal(True)

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

# REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

def seed_worker(worker_id):
	worker_seed = SEED + worker_id
	np.random.seed(worker_seed)
	random.seed(worker_seed)

# PATHS
ROOT_PATH = Path(r"D:\Anish\Research\UrbanSound8k")
AUDIO_PATH = ROOT_PATH / "audio"
METADATA_PATH = ROOT_PATH / "metadata" / "UrbanSound8K.csv"
CHECKPOINT_PATH = Path("./Checkpoint")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRAINING CONFIG
BATCH_SIZE = 32
MAX_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 3e-4
PATIENCE = 20

MIXUP_ALPHA = 0.2
MIXUP_STOP_EPOCH = 10

EMA_DECAY = 0.9997

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

# 4-Channel Feature Extractor for MPIMN
def extract_features(audio):
	# 1. Fix length to exactly 4 seconds
	if len(audio) > MAX_AUDIO_LENGTH:
		max_offset = len(audio) - MAX_AUDIO_LENGTH
		offset = np.random.randint(0, max_offset)
		audio = audio[offset:offset + MAX_AUDIO_LENGTH]
	else:
		audio = librosa.util.pad_center(data=audio, size=MAX_AUDIO_LENGTH)

	# 2. Log-Mel Spectrogram
	mel = librosa.feature.melspectrogram(
		y=audio,
		sr=SAMPLE_RATE,
		n_fft=N_FFT,
		hop_length=HOP_LENGTH,
		n_mels=N_MELS,
		power=2.0
	)

	logmel = librosa.power_to_db(mel, ref=np.max)

	# 3. Delta and Delta-Delta
	delta = librosa.feature.delta(logmel)
	delta2 = librosa.feature.delta(logmel, order=2)

	# 4. Spectral Contrast
	contrast = librosa.feature.spectral_contrast(
		y=audio,
		sr=SAMPLE_RATE,
		n_fft=N_FFT,
		hop_length=HOP_LENGTH
	)

	# contrast shape: (7 bands, T)
	# We upsample frequency dimension to 128
	contrast_resized = librosa.util.fix_length(
		data=contrast,
		size=logmel.shape[0],
		axis=0
	)

	# 5. Channel-wise Standardization (CRITICAL)
	def normalize(x):
		return (x - x.mean()) / (x.std() + 1e-6)

	logmel = normalize(logmel)
	delta = normalize(delta)
	delta2 = normalize(delta2)
	contrast_resized = normalize(contrast_resized)

	# 6. Stack to 4 Channels
	features = np.stack(
		[logmel, delta, delta2, contrast_resized],
		axis=0
	).astype(np.float32)

	return torch.tensor(features)

# AUGMENTATION
wave_aug = Compose([
	AddGaussianNoise(0.001, 0.01, p=0.4),
	TimeStretch(0.9, 1.1, p=0.3),
	PitchShift(-3, 3, p=0.3)
])

class SpecAugment(nn.Module):
	def __init__(self, freq_mask=24, time_mask=64):
		super().__init__()
		self.freq_mask = freq_mask
		self.time_mask = time_mask

	def forward(self, x, epoch):
		if not self.training:
			return x

		# Apply only first 60% epochs
		if epoch > int(MAX_EPOCHS * 0.6):
			return x

		B, C, F, T = x.shape

		for b in range(B):
			f = np.random.randint(0, self.freq_mask)
			t = np.random.randint(0, self.time_mask)
			f0 = np.random.randint(0, F - f)
			t0 = np.random.randint(0, T - t)

			x[b, :, f0:f0+f, :] = 0
			x[b, :, :, t0:t0+t] = 0

		return x

spec_aug = SpecAugment()

# DATASET
class UrbanSoundDataset(Dataset):
	def __init__(self, df, train=True):
		self.df = df.reset_index(drop=True)
		self.train = train

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		path = AUDIO_PATH / f"fold{row.fold}" / row.slice_file_name

		y, sr = librosa.load(path, sr=SAMPLE_RATE)

		y = librosa.util.fix_length(
			data=y, size=MAX_AUDIO_LENGTH
		)

		if self.train:
			y = wave_aug(samples=y, sample_rate=SAMPLE_RATE)

		x = extract_features(y)

		return x, int(row.classID)

		return x, int(row.classID)

# MIXUP
def mixup_data(x, y, alpha):
	lam = np.random.beta(alpha, alpha)
	lam = max(0.1, min(0.9, lam))
	index = torch.randperm(x.size(0)).to(x.device)
	mixed_x = lam * x + (1 - lam) * x[index]
	return mixed_x, y, y[index], lam

# EMA
class EMA:
	def __init__(self, model, decay):
		self.model = model
		self.decay = decay
		self.shadow = {}
		for name, param in model.named_parameters():
			if param.requires_grad:
				self.shadow[name] = param.data.clone()

	def update(self):
		for name, param in self.model.named_parameters():
			if param.requires_grad:
				new_average = (
					self.decay * self.shadow[name]
					+ (1.0 - self.decay) * param.data
				)
				self.shadow[name] = new_average.clone()

	def apply_shadow(self):
		for name, param in self.model.named_parameters():
			if param.requires_grad:
				param.data.copy_(self.shadow[name])

# Channel Gate
class ChannelGate(nn.Module):
	def __init__(self, dim, reduction=8):
		super().__init__()
		hidden = dim // reduction
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(dim, hidden, bias=False),
			nn.ReLU(),
			nn.Linear(hidden, dim, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.shape
		y = self.pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y

# DW Residual (expansion=1.5)
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

# Parallel Block
class ParallelBlock(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
		self.global_branch = nn.Conv2d(
			dim, dim, 5, padding=4, dilation=2,
			groups=dim, bias=False
		)
		self.fuse = nn.Conv2d(dim*2, dim, 1, bias=False)
		self.act = nn.GELU()
		self.gate = ChannelGate(dim)

	def forward(self, x):
		l = self.local(x)
		g = self.global_branch(x)
		x = torch.cat([l, g], dim=1)
		x = self.act(self.fuse(x))
		x = self.gate(x)
		return x

# MODEL
class Model(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(4, 64, 3, padding=1, bias=False),   # <-- changed from 1 to 4
			nn.BatchNorm2d(64),
			nn.GELU()
		)

		# Stage 1 (64)
		self.stage1 = nn.Sequential(
			ParallelBlock(64),
			DWRes(64)
		)

		self.down1 = nn.Conv2d(64, 128, 2, stride=2, bias=False)

		# Stage 2 (128)
		self.stage2 = nn.Sequential(
			ParallelBlock(128),
			DWRes(128),
			DWRes(128)
		)
		self.down2 = nn.Conv2d(128, 240, 2, stride=2, bias=False)

		# Stage 3 (240)
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

model = Model().to(DEVICE)

def run_epoch(model, loader, optimizer=None, scaler=None, ema=None, epoch=0, desc="Train"):
	train = optimizer is not None
	model.train() if train else model.eval()
	
	pbar = tqdm(loader, desc=desc, leave=False, ncols=0, unit="batch")

	total_loss = 0
	correct_samples = 0
	total_samples = 0
	
	preds, labels = [], []

	for x, y in pbar:
		x, y = x.to(DEVICE), y.to(DEVICE)

		if train and epoch <= MIXUP_STOP_EPOCH:
			x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)

		if train:
			x = spec_aug(x, epoch)

		with torch.amp.autocast('cuda', enabled=True):
			out = model(x)

			if train and epoch <= MIXUP_STOP_EPOCH:
				loss = (
					lam * loss_fn(out, y_a)
					+ (1 - lam) * loss_fn(out, y_b)
				)
			else:
				loss = loss_fn(out, y)

		if train:
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			ema.update()

		batch_size = x.size(0)
		total_loss += loss.item() * batch_size
		
		# Get predictions
		batch_preds = out.argmax(1)
		
		# Calculate running accuracy (compare vs original y)
		correct_samples += (batch_preds == y).sum().item()
		total_samples += batch_size
		
		running_acc = correct_samples / total_samples
		running_loss = total_loss / total_samples

		# Update TQDM bar with running metrics
		pbar.set_postfix({
			'loss': f"{running_loss:.4f}", 
			'acc': f"{running_acc:.4f}"
		})

		# Store for final F1 calculation
		preds.extend(batch_preds.detach().cpu().numpy())
		labels.extend(y.cpu().numpy())

	# Final metrics for the Epoch Log
	acc = accuracy_score(labels, preds)
	f1 = f1_score(labels, preds, average="macro")
	loss = total_loss / len(loader.dataset)

	return acc, f1, loss

# MAIN
def main():
	CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)
	LOG_FILE = CHECKPOINT_PATH / "training_log.json" # Path for the JSON log

	df = pd.read_csv(METADATA_PATH)

	train_df = df[df.fold.isin(TRAIN_FOLDS)]
	val_df = df[df.fold == VAL_FOLD]
	test_df = df[df.fold == TEST_FOLD]

	train_loader = DataLoader(
		UrbanSoundDataset(train_df, train=True),
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
		worker_init_fn=seed_worker
	)

	val_loader = DataLoader(
		UrbanSoundDataset(val_df, train=False),
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
		worker_init_fn=seed_worker
	)

	test_loader = DataLoader(
		UrbanSoundDataset(test_df, train=False),
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
		worker_init_fn=seed_worker
	)

	# INSERT MODEL HERE
	model = Model().to(DEVICE)

	global loss_fn
	loss_fn = nn.CrossEntropyLoss()

	optimizer = optim.AdamW(
		model.parameters(),
		lr=LR,
		weight_decay=WEIGHT_DECAY
	)

	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer,
		T_0=20,
		T_mult=2
	)

	scaler = GradScaler('cuda')
	ema = EMA(model, EMA_DECAY)

	best_f1 = 0
	patience_ctr = 0
	history = []

	print(f"Starting training on {DEVICE}...")

	for epoch in range(1, MAX_EPOCHS + 1):
		
		start_time = time.time()

		# --- Training ---
		tr_acc, tr_f1, tr_loss = run_epoch(
			model, train_loader,
			optimizer, scaler, ema, epoch, 
			desc="Train" 
		)

		# --- Validation ---
		ema.apply_shadow()
		va_acc, va_f1, va_loss = run_epoch(
			model, val_loader, epoch=epoch, 
			desc="Valid"
		)

		scheduler.step()

		# --- Checkpoint & Patience ---
		if va_f1 > best_f1:
			best_f1 = va_f1
			patience_ctr = 0
			torch.save(model.state_dict(), CHECKPOINT_PATH / "best.pth")
			status_msg = "Best Epoch"
		else:
			patience_ctr += 1
			status_msg = f"{patience_ctr} / {PATIENCE}"

		# --- Timing ---
		end_time = time.time() - start_time
		epoch_mins, epoch_secs = divmod(end_time, 60)
		time_str = f"{int(epoch_mins):02d}m {int(epoch_secs):02d}s"

		# --- JSON Logging ---
		epoch_log = {
			"epoch": epoch,
			"time": end_time,
			"train_loss": round(tr_loss, 4),
			"train_acc": round(tr_acc, 4),
			"train_f1": round(tr_f1, 4),
			"val_loss": round(va_loss, 4),
			"val_acc": round(va_acc, 4),
			"val_f1": round(va_f1, 4),
			"best_val_f1": round(best_f1, 4),
			"patience": patience_ctr
		}
		history.append(epoch_log)

		# Save to file immediately (overwrites previous file with updated list)
		with open(LOG_FILE, "w") as f:
			json.dump(history, f, indent=4)

		# --- Terminal Output ---
		print(f"Epoch-{epoch:<3d}  [{time_str}]     [{status_msg}]")
		print(f"TRN => Acc: {tr_acc:.4f}     F1: {tr_f1:.4f}     Loss: {tr_loss:.4f}")
		print(f"VAL => Acc: {va_acc:.4f}     F1: {va_f1:.4f}     Loss: {va_loss:.4f}")
		print("")

		if patience_ctr >= PATIENCE:
			print(f"Early stopping triggered at epoch {epoch}")
			break

	print(f"Best Validation F1: {best_f1:.4f}")

	# Load best model for testing
	model.load_state_dict(
		torch.load(CHECKPOINT_PATH / "best.pth")
	)

	te_acc, te_f1, te_loss = run_epoch(
		model, test_loader, desc="Test"
	)

	# use classification report for printing the testing classification report
	y_true, y_pred = [], []
	for x, y in tqdm(test_loader, desc="Test", ncols=0, leave=False, unit="batch"):
		x, y = x.to(DEVICE), y.to(DEVICE)
		with torch.no_grad():
			out = model(x)
		y_true.extend(y.cpu().numpy())
		y_pred.extend(out.argmax(1).cpu().numpy())
	
	# Optional: Log test results to the same JSON
	final_log = {
		"test_loss": round(te_loss, 4),
		"test_acc": round(te_acc, 4),
		"test_f1": round(te_f1, 4)
	}

	history.append(final_log)
	with open(LOG_FILE, "w") as f:
		json.dump(history, f, indent=4)

	print(classification_report(y_true, y_pred))

if __name__ == "__main__":
	main()