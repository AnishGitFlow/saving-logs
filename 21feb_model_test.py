import torch
import torch.nn as nn
import librosa
import numpy as np
import torchaudio.transforms as T
from pathlib import Path
import pandas as pd
import noisereduce as nr
from torch.utils.data import Dataset, DataLoader
import os, random
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from collections import OrderedDict

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

BATCH_SIZE = 32

SAMPLE_RATE = 22050
MAX_AUDIO_LENGTH_SEC = 4
MAX_AUDIO_LENGTH = SAMPLE_RATE * MAX_AUDIO_LENGTH_SEC

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
NUM_CLASSES = 10

CLASS_NAMES = [
	"air_conditioner", "car_horn", "children_playing", "dog_bark", 
	"drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

ROOT_PATH = Path(r"D:\Anish\Research\UrbanSound8k")
AUDIO_PATH = ROOT_PATH / "audio"
METADATA_PATH = ROOT_PATH / "metadata" / "UrbanSound8K.csv"
CHECKPOINT_PATH = Path("21feb")

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
	def __init__(self, num_classes=10, drop_path_rate=0.2):
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

def extract_features(audio):
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
		[norm(logmel), norm(delta), norm(delta2), norm(contrast)],
		axis=0
	).astype(np.float32)

	return torch.tensor(features)

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

		x = extract_features(y)
		return x, int(row.classID)

def test(model, loader, loss_fn):
	model.eval()
	total_loss, total_correct, total = 0, 0, 0
	preds, labels = [], []

	pbar = tqdm(loader, desc="Test", leave=False, unit="batch", ncols=0)

	with torch.no_grad():
		for x, y in pbar:
			x, y  = x.to(DEVICE), y.to(DEVICE)
			out   = model(x)
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
	report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)

	return acc, f1, loss, report

def main():
	df       = pd.read_csv(METADATA_PATH)
	train_df = df[df.fold.isin([1,2,3,4,5,6,7,8])]
	test_df  = df[df.fold == 10]

	test_loader = DataLoader(
		UrbanSoundDataset(test_df,  False), batch_size=BATCH_SIZE,
		shuffle=False, num_workers=2, worker_init_fn=worker_fn,
		generator=g, prefetch_factor=1
	)

	model = Model(drop_path_rate=0.2).to(DEVICE)
	model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Total Model Parameters: {model_total_params:,}")
	print(f"Using model {CHECKPOINT_PATH}/{CHECKPOINT_PATH.name}_best.pth")

	class_weights = compute_class_weights(train_df, NUM_CLASSES).to(DEVICE)
	loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

	ckpt = torch.load(CHECKPOINT_PATH / f"{CHECKPOINT_PATH.name}_best.pth", weights_only=False)
	model.load_state_dict(ckpt['model_state_dict'])

	te_acc, te_f1, te_loss, report = test(model, test_loader, loss_fn)
	print(f"TEST => Acc: {te_acc:.4f}   F1: {te_f1:.4f}   Loss: {te_loss:.4f}\n")
	print(report)

if __name__ == "__main__":
	main()
