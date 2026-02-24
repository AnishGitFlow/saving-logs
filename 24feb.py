import os
import random
import time
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

import librosa
import noisereduce as nr
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator
from sklearn.metrics import accuracy_score, classification_report, f1_score

torch.set_num_threads(6)
torch.backends.cudnn.enabled = False

os.environ["PYTHONHASHSEED"] = str(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

def worker_fn(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_PATH = Path(r"D:\Anish\Research\UrbanSound8k")
AUDIO_PATH = ROOT_PATH / "audio"
METADATA_PATH = ROOT_PATH / "metadata" / "UrbanSound8K.csv"
CHECKPOINT_PATH = Path(Path(__file__).stem)
CHECKPOINT_PATH.mkdir(exist_ok=True)

BATCH_SIZE = 32
MAX_EPOCHS = 100                # [CHANGE 1] More epochs: cosine decay needs room to breathe; early stopping will cut it short if needed anyway
LR = 3e-4
WEIGHT_DECAY = 2e-2
PATIENCE = 25                   # [CHANGE 2] Increased from 20 → 25 to pair with more epochs

WARMUP_EPOCHS = 5               # [CHANGE 3] Linear LR warmup before cosine decay kicks in

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

CLASS_NAMES = [
	"air_conditioner", "car_horn", "children_playing", "dog_bark", 
	"drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

SIREN_ID = CLASS_TO_ID["siren"]
CAR_HORN_ID = CLASS_TO_ID["car_horn"]
GUNSHOT_ID = CLASS_TO_ID["gun_shot"]
CHILDREN_ID = CLASS_TO_ID["children_playing"]
STREET_ID = CLASS_TO_ID["street_music"]

def compute_class_weights(df, num_classes):
	counts = df["classID"].value_counts().sort_index()
	total = len(df)
	weights = torch.tensor(
		[total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
		dtype=torch.float32
	)
	return weights / weights.mean()

def drop_path(x, drop_prob: float = 0., training: bool = False):
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)
	random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	random_tensor.floor_()
	output = x.div(keep_prob) * random_tensor
	return output

class DropPath(nn.Module):
	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training) # type: ignore

def extract_features(audio, epoch, is_train):
	if len(audio) > MAX_AUDIO_LENGTH:
		audio = audio[:MAX_AUDIO_LENGTH]
	else:
		audio = librosa.util.pad_center(audio, size=MAX_AUDIO_LENGTH)

	mel = librosa.feature.melspectrogram(
		y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
		hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
	)
	logmel = librosa.power_to_db(mel, ref=np.max)

	delta  = librosa.feature.delta(logmel)
	delta2 = librosa.feature.delta(logmel, order=2)

	logmel_aug = logmel.copy()
	if is_train:
		f_p, t_p = 24, 48
		if epoch >= MAX_EPOCHS * 0.5: f_p, t_p = 12, 24
		if epoch >= MAX_EPOCHS * 0.8: f_p, t_p = 0,  0

		if f_p > 0:
			t = torch.tensor(logmel_aug).unsqueeze(0)
			aug_f = T.FrequencyMasking(freq_mask_param=f_p)
			aug_t = T.TimeMasking(time_mask_param=t_p)
			t = aug_f(t)
			t = aug_t(t)
			logmel_aug = t.squeeze(0).numpy()

	contrast = librosa.feature.spectral_contrast(
		y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
	)
	contrast_tensor = torch.tensor(contrast).unsqueeze(0).unsqueeze(0)
	target_h, target_w = logmel.shape
	contrast_tensor = torch.nn.functional.interpolate(
		contrast_tensor, size=(target_h, target_w),
		mode='bilinear', align_corners=False
	)
	contrast = contrast_tensor.squeeze().numpy()

	def norm(x):
		return (x - x.mean()) / (x.std() + 1e-6)

	features = np.stack(
		[norm(logmel_aug), norm(delta), norm(delta2), norm(contrast)],
		axis=0
	).astype(np.float32)

	return torch.tensor(features)

wave_aug_general = Compose([
    AddGaussianNoise(0.001, 0.01, p=0.4),
    TimeStretch(0.9, 1.1, p=0.3),
])

wave_aug_pitch = PitchShift(-3, 3, p=0.4)

wave_aug_room = RoomSimulator(p=0.4)

class UrbanSoundDataset(Dataset):
	def __init__(self, df: pd.DataFrame, train: bool = True):
		self.df    = df.reset_index(drop=True)
		self.train = train
		self.epoch = 0

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		row  = self.df.iloc[idx]
		path = AUDIO_PATH / f"fold{row.fold}" / row.slice_file_name

		y_orig, _ = librosa.load(path, sr=SAMPLE_RATE)
		y_orig_len_fix = librosa.util.fix_length(y_orig, size=MAX_AUDIO_LENGTH)

		y = nr.reduce_noise(y=y_orig_len_fix, sr=SAMPLE_RATE)

		class_id = int(row.classID)

		if self.train:
			y = wave_aug_general(samples=y, sample_rate=SAMPLE_RATE)

			if class_id not in [SIREN_ID, CAR_HORN_ID, GUNSHOT_ID]:
				y = wave_aug_pitch(samples=y, sample_rate=SAMPLE_RATE)

			if class_id not in [CHILDREN_ID, STREET_ID]:
				y = wave_aug_room(samples=y, sample_rate=SAMPLE_RATE)

		x = extract_features(y, self.epoch, is_train=self.train)
		return x, int(row.classID)

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

class SpatialGate(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = x.mean(dim=1, keepdim=True)
		max_out = x.max(dim=1, keepdim=True).values
		y = torch.cat([avg_out, max_out], dim=1)
		y = self.sigmoid(self.conv(y))
		return x * y

class DWRes(nn.Module):
	def __init__(self, dim, drop_path=0.0):
		super().__init__()
		hidden = int(dim * 1.5)
		self.dw   = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
		self.pw   = nn.Conv2d(dim, hidden, 1, bias=False)
		self.act  = nn.GELU()
		self.proj = nn.Conv2d(hidden, dim, 1, bias=False)
		self.bn   = nn.BatchNorm2d(dim)
		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

	def forward(self, x):
		r = x
		x = self.dw(x)
		x = self.act(self.pw(x))
		x = self.proj(x)
		x = self.bn(x)
		return r + self.drop_path(x)

class ParallelBlock(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.local         = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
		self.global_branch = nn.Conv2d(dim, dim, 5, padding=4, dilation=2, groups=dim, bias=False)
		self.fuse          = nn.Conv2d(dim * 2, dim, 1, bias=False)
		self.act           = nn.GELU()
		self.channel_gate  = ChannelGate(dim)
		self.spatial_gate  = SpatialGate()

	def forward(self, x):
		l = self.local(x)
		g = self.global_branch(x)
		x = torch.cat([l, g], dim=1)
		x = self.act(self.fuse(x))
		x = self.channel_gate(x)
		x = self.spatial_gate(x)
		return x

class Model(nn.Module):
	def __init__(self, num_classes=10, drop_path_rate=0.15):
		super().__init__()

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 9)]

		self.stem = nn.Sequential(
			nn.Conv2d(4, 64, 3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.GELU()
		)

		self.stage1 = nn.Sequential(ParallelBlock(64),  DWRes(64,  drop_path=dpr[0]))
		self.down1  = nn.Conv2d(64, 128, 2, stride=2, bias=False)

		self.stage2 = nn.Sequential(
			ParallelBlock(128),
			DWRes(128, drop_path=dpr[1]),
			DWRes(128, drop_path=dpr[2])
		)
		self.down2 = nn.Conv2d(128, 240, 2, stride=2, bias=False)

		self.stage3 = nn.Sequential(
			ParallelBlock(240),
			DWRes(240, drop_path=dpr[3]),
			DWRes(240, drop_path=dpr[4]),
			DWRes(240, drop_path=dpr[5])
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

def mixup_data(x, y, alpha=1.0):
	lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
	index = torch.randperm(x.size(0), device=DEVICE)
	mixed_x = lam * x + (1 - lam) * x[index]
	return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def build_scheduler(optimizer, warmup_epochs, total_epochs):
	def lr_lambda(epoch):
		if epoch < warmup_epochs:
			return (epoch + 1) / warmup_epochs          # linear ramp 0 → 1
		progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
		return 0.01 + 0.5 * (1 - 0.01) * (1 + np.cos(np.pi * progress))

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, loss_fn, epoch):
	model.train()
	total_loss, total_correct, total = 0, 0, 0
	preds, labels = [], []

	pbar = tqdm(loader, desc=f"Epoch-{epoch:02d} Train", leave=False, unit="batch", ncols=0)

	for x, y in pbar:
		x, y = x.to(DEVICE), y.to(DEVICE)
		optimizer.zero_grad()

		if epoch <= 15:
			mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.25)
			out  = model(mixed_x)
			loss = mixup_criterion(loss_fn, out, y_a, y_b, lam)
		else:
			out  = model(x)
			loss = loss_fn(out, y)

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

		optimizer.step()

		bs = x.size(0)
		total_loss    += loss.item() * bs
		pred           = out.argmax(1)
		total_correct += (pred == y).sum().item()
		total         += bs

		preds.extend(pred.detach().cpu().numpy())
		labels.extend(y.detach().cpu().numpy())

		pbar.set_postfix(OrderedDict([
			("acc",  f"{total_correct/total:.4f}"),
			("loss", f"{total_loss/total:.4f}")
		]))

	return accuracy_score(labels, preds), f1_score(labels, preds, average="macro"), total_loss / total

def validate(model, loader, loss_fn, epoch, return_report=False):
	model.eval()
	total_loss, total_correct, total = 0, 0, 0
	preds, labels = [], []
	
	def forward_tta(model, x):
		outputs = []

		outputs.append(model(x))

		x_shift = torch.roll(x, shifts=10, dims=3)
		outputs.append(model(x_shift))

		x_amp = x * 1.02
		outputs.append(model(x_amp))
		return torch.mean(torch.stack(outputs), dim=0)

	desc = f"Epoch-{epoch:02d} Valid" if epoch >= 1 else "Test"
	pbar = tqdm(loader, desc=desc, leave=False, unit="batch", ncols=0)

	with torch.no_grad():
		for x, y in pbar:
			x, y  = x.to(DEVICE), y.to(DEVICE)
			out   = forward_tta(model, x)
			loss  = loss_fn(out, y)

			bs = x.size(0)
			total_loss    += loss.item() * bs
			pred           = out.argmax(1)
			total_correct += (pred == y).sum().item()
			total         += bs

			preds.extend(pred.cpu().numpy())
			labels.extend(y.cpu().numpy())

			pbar.set_postfix(OrderedDict([
				("acc",  f"{total_correct/total:.4f}"),
				("loss", f"{total_loss/total:.4f}")
			]))

	acc    = accuracy_score(labels, preds)
	f1     = f1_score(labels, preds, average="macro")
	loss   = total_loss / total
	report = classification_report(labels, preds, target_names=CLASS_NAMES) if return_report else ""

	return acc, f1, loss, report

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def main():
	df       = pd.read_csv(METADATA_PATH)
	train_df = df[df.fold.isin(TRAIN_FOLDS)]
	val_df   = df[df.fold == VAL_FOLD]
	test_df  = df[df.fold == TEST_FOLD]

	train_dataset = UrbanSoundDataset(train_df, True)
	val_dataset   = UrbanSoundDataset(val_df,   False)
	test_dataset  = UrbanSoundDataset(test_df,  False)

	train_loader = DataLoader(
		train_dataset, batch_size=BATCH_SIZE, shuffle=True,
		num_workers=2, worker_init_fn=worker_fn, generator=g, prefetch_factor=2
	)
	val_loader = DataLoader(
		val_dataset, batch_size=BATCH_SIZE, shuffle=False,
		num_workers=2, worker_init_fn=worker_fn, generator=g, prefetch_factor=1
	)
	test_loader = DataLoader(
		test_dataset, batch_size=BATCH_SIZE, shuffle=False,
		num_workers=2, worker_init_fn=worker_fn, generator=g, prefetch_factor=1
	)

	model = Model(drop_path_rate=0.15).to(DEVICE)
	model_total_params = sum(p.numel() for p in model.parameters())
	print(f"Total Model Parameters: {model_total_params:,}\n")

	class_weights = compute_class_weights(train_df, NUM_CLASSES).to(DEVICE)
	loss_fn = FocalLoss(weight=class_weights, gamma=1.5)
	optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
	scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)

	history  = []
	log_file = CHECKPOINT_PATH / f"{CHECKPOINT_PATH.name}_log.json"

	best_f1          = 0
	patience_counter = 0

	try:
		for epoch in range(1, MAX_EPOCHS + 1):
			train_loader.dataset.epoch = epoch

			start_time = time.perf_counter()

			tr_acc, tr_f1, tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch)
			va_acc, va_f1, va_loss, _ = validate(model, val_loader, loss_fn, epoch)

			scheduler.step()
			current_lr = scheduler.get_last_lr()[0]

			status_msg = ""

			if va_f1 > best_f1:
				best_f1          = va_f1
				patience_counter = 0
				torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'val_f1': best_f1
				}, CHECKPOINT_PATH / f"{CHECKPOINT_PATH.name}_best.pth")
				status_msg = "⁕ Best Epoch"
			else:
				patience_counter += 1
				status_msg = f"{patience_counter} / {PATIENCE}"
				if patience_counter >= PATIENCE:
					print(f"Early stopping at epoch {epoch}")
					break

			elapsed = time.perf_counter() - start_time
			m, s    = divmod(int(elapsed), 60)

			epoch_log = OrderedDict([
				("epoch",     epoch),
				("train_acc", tr_acc),
				("train_f1",  tr_f1),
				("train_loss",tr_loss),
				("val_acc",   va_acc),
				("val_f1",    va_f1),
				("val_loss",  va_loss),
				("lr",        current_lr),
				("duration",  elapsed)
			])
			history.append(epoch_log)
			with open(log_file, "w") as f:
				json.dump(history, f, indent=4)

			print(f"Epoch-{epoch:<2d}   {m:02d}:{s:02d}   LR: {current_lr:.2e}   {status_msg}")
			print(f"TRN => Acc: {tr_acc:.4f}   F1: {tr_f1:.4f}   Loss: {tr_loss:.4f}")
			print(f"VAL => Acc: {va_acc:.4f}   F1: {va_f1:.4f}   Loss: {va_loss:.4f}\n")

	except KeyboardInterrupt:
		print("\n\nTraining interrupted, exiting...")
		exit(0)

	ckpt = torch.load(CHECKPOINT_PATH / f"{CHECKPOINT_PATH.name}_best.pth", weights_only=False)
	model.load_state_dict(ckpt['model_state_dict'])

	te_acc, te_f1, te_loss, report = validate(model, test_loader, loss_fn, 0, return_report=True)
	print(f"TEST => Acc: {te_acc:.4f}   F1: {te_f1:.4f}   Loss: {te_loss:.4f}\n")
	print(report)

if __name__ == "__main__":
	main()
