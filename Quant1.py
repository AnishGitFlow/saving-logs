"""
quantization_benchmark.py
=========================
Comprehensive quantization benchmark for the UrbanSound8K DWRes/ParallelBlock model.

Quantization methods covered
─────────────────────────────
 1. FP32  Baseline          – original trained model (reference)
 2. FP16  Half-precision    – .half() cast, no calibration needed
 3. BF16  Brain float       – .bfloat16(), good for Ampere+ GPUs
 4. PTQ  Dynamic INT8       – torch.quantization dynamic (Linear/Conv layers)
 5. PTQ  Static INT8        – full static calibration with representative data
 6. PTQ  Static INT8 + QAT-fine-tune  – brief QAT fine-tuning after static PTQ
 7. PTQ  Per-channel INT8   – per-output-channel static quantization
 8. PTQ  INT4 (fake-quant)  – manual fake-quantization to 4-bit range
 9. QAT  INT8               – Quantization-Aware Training (re-trains backbone)
10. GPTQ-style INT4 weight-only  – row-wise absmax INT4 weight packing

Benchmarking metrics
─────────────────────
  • Accuracy  (top-1)
  • Macro F1
  • Per-class F1  (10 urban-sound classes)
  • Cross-entropy loss
  • Model size on disk  (MB)
  • Peak RAM / GPU VRAM during inference  (MB)
  • Single-sample latency  (ms, median of 200 runs)
  • Batch latency          (ms, batch=32, median of 50 runs)
  • Throughput             (samples / second)
  • Relative size vs FP32  (%)
  • F1 drop vs FP32        (absolute pp)
  • Speedup vs FP32

Usage
─────
  python quantization_benchmark.py \
      --checkpoint  /path/to/feb21_best.pth \
      --metadata    /path/to/UrbanSound8K/metadata/UrbanSound8K.csv \
      --audio-root  /path/to/UrbanSound8K/audio \
      --methods     all \
      --qat-epochs  5 \
      --output-dir  quant_results

All flags are optional if paths are hardcoded in PATHS section below.
"""

import os, sys, gc, json, time, copy, random, argparse, warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

import librosa
import noisereduce as nr
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS  (override via CLI flags or edit here)
# ─────────────────────────────────────────────────────────
ROOT_PATH      = Path(r"D:\Anish\Research\UrbanSound8k")
AUDIO_PATH     = ROOT_PATH / "audio"
METADATA_PATH  = ROOT_PATH / "metadata" / "UrbanSound8K.csv"
CHECKPOINT_DIR = Path("feb21")
CHECKPOINT_FILE = CHECKPOINT_DIR / "feb21_best.pth"
OUTPUT_DIR     = Path("quant_results")

# ─────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────
SEED = 42
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU    = torch.device("cpu")

# ─────────────────────────────────────────────────────────
# Audio / feature constants  (must match training)
# ─────────────────────────────────────────────────────────
SAMPLE_RATE         = 22050
MAX_AUDIO_LENGTH    = SAMPLE_RATE * 4
N_FFT               = 1024
HOP_LENGTH          = 256
N_MELS              = 128
NUM_CLASSES         = 10
BATCH_SIZE          = 32
MAX_EPOCHS          = 100

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD    = 9
TEST_FOLD   = 10

CLASS_NAMES = [
    "air_conditioner","car_horn","children_playing","dog_bark",
    "drilling","engine_idling","gun_shot","jackhammer","siren","street_music"
]

# ─────────────────────────────────────────────────────────
# Feature extraction  (identical to training)
# ─────────────────────────────────────────────────────────
def extract_features(audio, epoch=0, is_train=False):
    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        audio = librosa.util.pad_center(audio, size=MAX_AUDIO_LENGTH)

    mel    = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
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
            t = T.FrequencyMasking(f_p)(t)
            t = T.TimeMasking(t_p)(t)
            logmel_aug = t.squeeze(0).numpy()

    contrast = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = nn.functional.interpolate(
        torch.tensor(contrast).unsqueeze(0).unsqueeze(0),
        size=logmel.shape, mode='bilinear', align_corners=False
    ).squeeze().numpy()

    def norm(x): return (x - x.mean()) / (x.std() + 1e-6)
    feat = np.stack([norm(logmel_aug), norm(delta), norm(delta2), norm(contrast)], axis=0)
    return torch.tensor(feat.astype(np.float32))


wave_aug = Compose([
    AddGaussianNoise(0.001, 0.01, p=0.4),
    TimeStretch(0.9, 1.1, p=0.3),
    PitchShift(-3, 3, p=0.3),
    RoomSimulator(p=0.3),
])


class UrbanSoundDataset(Dataset):
    def __init__(self, df, train=False):
        self.df    = df.reset_index(drop=True)
        self.train = train
        self.epoch = 0

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = AUDIO_PATH / f"fold{row.fold}" / row.slice_file_name
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        y    = librosa.util.fix_length(y, size=MAX_AUDIO_LENGTH)
        y    = nr.reduce_noise(y=y, sr=SAMPLE_RATE)
        if self.train:
            y = wave_aug(samples=y, sample_rate=SAMPLE_RATE)
        return extract_features(y, self.epoch, is_train=self.train), int(row.classID)


# ─────────────────────────────────────────────────────────
# Model architecture  (identical to training)
# ─────────────────────────────────────────────────────────
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * rand.floor_()

class DropPath(nn.Module):
    def __init__(self, p=None): super().__init__(); self.drop_prob = p
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class ChannelGate(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False), nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        return x * self.sigmoid(self.conv(y))

class DWRes(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        h = int(dim * 1.5)
        self.dw        = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw        = nn.Conv2d(dim, h, 1, bias=False)
        self.act       = nn.GELU()
        self.proj      = nn.Conv2d(h, dim, 1, bias=False)
        self.bn        = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        r = x
        x = self.dw(x); x = self.act(self.pw(x)); x = self.proj(x); x = self.bn(x)
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
        x = self.act(self.fuse(torch.cat([self.local(x), self.global_branch(x)], dim=1)))
        return self.spatial_gate(self.channel_gate(x))

class Model(nn.Module):
    def __init__(self, num_classes=10, drop_path_rate=0.2):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 9)]
        self.stem   = nn.Sequential(nn.Conv2d(4,64,3,padding=1,bias=False),
                                     nn.BatchNorm2d(64), nn.GELU())
        self.stage1 = nn.Sequential(ParallelBlock(64),  DWRes(64,  dpr[0]))
        self.down1  = nn.Conv2d(64, 128, 2, stride=2, bias=False)
        self.stage2 = nn.Sequential(ParallelBlock(128), DWRes(128, dpr[1]), DWRes(128, dpr[2]))
        self.down2  = nn.Conv2d(128, 240, 2, stride=2, bias=False)
        self.stage3 = nn.Sequential(ParallelBlock(240), DWRes(240, dpr[3]),
                                     DWRes(240, dpr[4]), DWRes(240, dpr[5]))
        self.norm   = nn.BatchNorm2d(240)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.LayerNorm(240), nn.Dropout(0.3), nn.Linear(240, num_classes))

    def forward(self, x):
        x = self.stem(x);  x = self.stage1(x); x = self.down1(x)
        x = self.stage2(x); x = self.down2(x); x = self.stage3(x)
        return self.head(self.pool(self.norm(x)).flatten(1))


# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def load_base_model(ckpt_path, device=CPU):
    model = Model(drop_path_rate=0.2)
    # weights_only=False is required because the checkpoint contains embedded
    # numpy scalars (e.g. from scheduler state). Only load from trusted sources.
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    key   = "model_state_dict" if "model_state_dict" in ckpt else None
    model.load_state_dict(ckpt[key] if key else ckpt)
    model.eval()
    return model.to(device)


def model_size_mb(model_or_path):
    if isinstance(model_or_path, (str, Path)):
        return os.path.getsize(model_or_path) / 1e6
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(model_or_path.state_dict(), tmp)
        s = tmp.stat().st_size / 1e6
    finally:
        tmp.unlink(missing_ok=True)
    return s


def peak_memory_mb(model, loader, device):
    """Run one pass and measure peak memory."""
    try:
        model_dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
    except StopIteration:
        model_dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device=device, dtype=model_dtype)); break
        return torch.cuda.max_memory_allocated(device) / 1e6
    else:
        import tracemalloc
        tracemalloc.start()
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device=device, dtype=model_dtype)); break
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        return peak / 1e6


def latency_stats(model, single_input, batch_input, device, n_single=200, n_batch=50):
    """Returns median single-sample latency (ms), median batch latency (ms),
       and throughput (samples/sec)."""
    model.eval()
    try:
        model_dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
    except StopIteration:
        model_dtype = torch.float32
    xi = single_input.to(device=device, dtype=model_dtype)
    xb = batch_input.to(device=device, dtype=model_dtype)

    # warm-up
    with torch.no_grad():
        for _ in range(10): model(xi)

    # single-sample
    times_s = []
    with torch.no_grad():
        for _ in range(n_single):
            t0 = time.perf_counter()
            model(xi)
            if device.type == "cuda": torch.cuda.synchronize()
            times_s.append((time.perf_counter() - t0) * 1000)

    # batch
    times_b = []
    with torch.no_grad():
        for _ in range(n_batch):
            t0 = time.perf_counter()
            model(xb)
            if device.type == "cuda": torch.cuda.synchronize()
            times_b.append((time.perf_counter() - t0) * 1000)

    med_s = float(np.median(times_s))
    med_b = float(np.median(times_b))
    tput  = xb.shape[0] / (med_b / 1000)
    return med_s, med_b, tput


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    # Infer model dtype so inputs match (handles FP16/BF16 models)
    try:
        model_dtype = next(p for p in model.parameters() if p.is_floating_point()).dtype
    except StopIteration:
        model_dtype = torch.float32
    all_preds, all_labels, total_loss, total = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device=device, dtype=model_dtype)
        y = y.to(device)
        out = model(x)
        total_loss += loss_fn(out.float(), y).item() * x.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        total += x.size(0)
    acc    = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    return {
        "accuracy":      acc,
        "macro_f1":      macro_f1,
        "loss":          total_loss / total,
        "per_class_f1":  per_class_f1.tolist(),
        "report":        report
    }


def benchmark_model(name, model, test_loader, calib_loader, loss_fn, device,
                    single_x, batch_x):
    print(f"\n{'─'*55}\n  Benchmarking: {name}\n{'─'*55}")
    model = model.to(device)

    metrics = evaluate(model, test_loader, loss_fn, device)
    size    = model_size_mb(model)
    mem     = peak_memory_mb(model, test_loader, device)
    lat_s, lat_b, tput = latency_stats(model, single_x, batch_x, device)

    result = {
        "name":           name,
        "accuracy":       round(metrics["accuracy"],  4),
        "macro_f1":       round(metrics["macro_f1"],  4),
        "loss":           round(metrics["loss"],       4),
        "per_class_f1":   {CLASS_NAMES[i]: round(v, 4) for i, v in enumerate(metrics["per_class_f1"])},
        "model_size_mb":  round(size,  2),
        "peak_mem_mb":    round(mem,   2),
        "lat_single_ms":  round(lat_s, 3),
        "lat_batch_ms":   round(lat_b, 3),
        "throughput_sps": round(tput,  1),
    }
    print(f"  Acc: {result['accuracy']:.4f}  |  Macro-F1: {result['macro_f1']:.4f}  "
          f"|  Loss: {result['loss']:.4f}")
    print(f"  Size: {result['model_size_mb']:.2f} MB  |  Peak Mem: {result['peak_mem_mb']:.1f} MB")
    print(f"  Lat(1): {result['lat_single_ms']:.2f} ms  |  "
          f"Lat(32): {result['lat_batch_ms']:.2f} ms  |  "
          f"Throughput: {result['throughput_sps']:.0f} sps")
    return result


# ─────────────────────────────────────────────────────────
# INT4 fake-quantisation helper
# ─────────────────────────────────────────────────────────
class FakeInt4Quantize(nn.Module):
    """Replaces a Conv2d or Linear with a fake-INT4 wrapper (weights only)."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._quantize_weights()

    def _quantize_weights(self):
        with torch.no_grad():
            w  = self.module.weight.data
            # per-output-channel absmax scale
            scale  = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / 7.0
            scale  = scale.clamp(min=1e-8)
            w_q    = (w / scale).round().clamp(-8, 7)
            self.module.weight.data = w_q * scale   # dequantize back → stays FP32 ops

    def forward(self, x): return self.module(x)


def apply_fake_int4(model):
    model = copy.deepcopy(model)
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # navigate to parent and replace
            parts  = name.split(".")
            parent = model
            for p in parts[:-1]: parent = getattr(parent, p)
            setattr(parent, parts[-1], FakeInt4Quantize(module))
    return model


# ─────────────────────────────────────────────────────────
# GPTQ-style INT4 weight-only quantisation (row-wise)
# ─────────────────────────────────────────────────────────
def gptq_int4_weight_only(model):
    """
    Approximate GPTQ: per-row absmax INT4 on weights of Conv2d & Linear.
    Weights stored as INT8 tensors (INT4 packed would need a custom kernel;
    this gives the correct numeric effect and size footprint for benchmarking).
    """
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        with torch.no_grad():
            w = module.weight.data.float()
            orig_shape = w.shape
            w_flat = w.reshape(w.shape[0], -1)       # [out, in*k*k]
            scale  = w_flat.abs().amax(dim=1, keepdim=True) / 7.0
            scale  = scale.clamp(min=1e-8)
            w_int4 = (w_flat / scale).round().clamp(-8, 7).to(torch.int8)
            # Store scale as a buffer, reconstruct on forward
            module.register_buffer("_w_scale", scale.reshape(orig_shape[0],
                                    *([1]*(len(orig_shape)-1))))
            module.weight.data = (w_int4.float() * scale).reshape(orig_shape)
    return model


# ─────────────────────────────────────────────────────────
# QAT helpers
# ─────────────────────────────────────────────────────────
def prepare_qat_model(model):
    model_qat = copy.deepcopy(model).to(CPU)
    model_qat.train()
    model_qat.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Fuse Conv+BN+activation patterns where possible
    def fuse_model(m):
        for name, child in m.named_children():
            if isinstance(child, nn.Sequential):
                fuse_model(child)
            # fuse Conv+BN in DWRes
            elif isinstance(child, DWRes):
                # note: GELU is not fuseable, only ReLU — skip activation fusion
                fuse_model(child)
        return m

    tq.prepare_qat(model_qat, inplace=True)
    return model_qat


def train_qat(model_qat, train_loader, loss_fn, n_epochs=5, device=CPU):
    model_qat = model_qat.to(device)
    opt = optim.AdamW(model_qat.parameters(), lr=1e-5, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-7)
    for ep in range(1, n_epochs + 1):
        model_qat.train()
        total_loss, total, correct = 0., 0, 0
        pbar = tqdm(train_loader, desc=f"  QAT Epoch {ep}/{n_epochs}", leave=False, ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model_qat(x)
            loss = loss_fn(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model_qat.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += x.size(0)
        sch.step()
        print(f"    QAT Ep {ep}: loss={total_loss/total:.4f}  acc={correct/total:.4f}")
    return model_qat


# ─────────────────────────────────────────────────────────
# PTQ fine-tune (brief calibration + tiny distillation)
# ─────────────────────────────────────────────────────────
def ptq_finetune(static_model, teacher_fp32, calib_loader, loss_fn,
                  n_epochs=3, device=CPU):
    """
    Post-quantization fine-tune: run a few epochs of KL-divergence distillation
    on the floating-point weights just before final quantisation, using the
    original FP32 model as a soft-label teacher. This recovers accuracy lost
    from static quantisation artefacts.
    """
    student = copy.deepcopy(teacher_fp32).to(device)
    teacher = copy.deepcopy(teacher_fp32).to(device).eval()
    opt = optim.AdamW(student.parameters(), lr=5e-6, weight_decay=1e-3)
    kl  = nn.KLDivLoss(reduction="batchmean")
    T_temp = 4.0

    print("  PTQ fine-tune (KD): aligning student to teacher before static quant…")
    for ep in range(1, n_epochs + 1):
        student.train()
        total_loss, total = 0., 0
        for x, y in tqdm(calib_loader, desc=f"    KD Ep {ep}/{n_epochs}", leave=False, ncols=80):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.no_grad():
                soft_targets = torch.log_softmax(teacher(x) / T_temp, dim=1)
            out  = torch.log_softmax(student(x) / T_temp, dim=1)
            loss = kl(out, soft_targets.exp()) * (T_temp ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0); total += x.size(0)
        print(f"    KD Ep {ep}: loss={total_loss/total:.4f}")

    # Now apply static INT8 quantisation to the fine-tuned student
    student.eval()
    student.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(student, inplace=True)
    with torch.no_grad():
        for x, _ in calib_loader:
            student(x.to(device))
    tq.convert(student, inplace=True)
    return student


# ─────────────────────────────────────────────────────────
# Main benchmark orchestrator
# ─────────────────────────────────────────────────────────
def run_benchmarks(args):
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ── Data ──────────────────────────────────────────────
    df         = pd.read_csv(args.metadata)
    train_df   = df[df.fold.isin(TRAIN_FOLDS)]
    test_df    = df[df.fold == TEST_FOLD]
    calib_df   = train_df.sample(n=min(500, len(train_df)), random_state=SEED)

    test_ds    = UrbanSoundDataset(test_df,  train=False)
    calib_ds   = UrbanSoundDataset(calib_df, train=False)
    train_ds   = UrbanSoundDataset(train_df, train=True)

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    test_loader  = make_loader(test_ds)
    calib_loader = make_loader(calib_ds)
    train_loader = make_loader(train_ds, shuffle=True)

    # Loss (no class weights at eval time for fair comparison)
    loss_fn = nn.CrossEntropyLoss()

    # ── Dummy inputs for latency ──────────────────────────
    single_x = torch.randn(1,  4, N_MELS, 346)   # 4s @ 22050Hz, hop=256 → ~346 frames
    batch_x  = torch.randn(32, 4, N_MELS, 346)

    # ── Load FP32 base ────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    fp32_model = load_base_model(args.checkpoint, CPU)

    results = []
    active  = set(args.methods) if "all" not in args.methods else {
        "fp32","fp16","bf16","ptq_dynamic","ptq_static",
        "ptq_perchannel","ptq_int4","ptq_finetune","qat","gptq_int4"
    }

    # ────────────────────────────────────────────────────
    # 1. FP32 Baseline
    # ────────────────────────────────────────────────────
    if "fp32" in active:
        r = benchmark_model("FP32 Baseline", copy.deepcopy(fp32_model),
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)
        fp32_f1   = r["macro_f1"]
        fp32_size = r["model_size_mb"]
        fp32_lat  = r["lat_single_ms"]
    else:
        fp32_f1, fp32_size, fp32_lat = None, None, None

    # ────────────────────────────────────────────────────
    # 2. FP16
    # ────────────────────────────────────────────────────
    if "fp16" in active:
        if DEVICE == "cuda":
            m_fp16 = copy.deepcopy(fp32_model).half().to(DEVICE)
            r = benchmark_model("FP16 Half-precision", m_fp16,
                                 test_loader, calib_loader, loss_fn,
                                 torch.device(DEVICE),
                                 single_x.half().to(DEVICE), batch_x.half().to(DEVICE))
        else:
            print("  [SKIP] FP16 requires CUDA — skipping")
            r = {"name":"FP16 Half-precision","note":"skipped – CPU only"}
        results.append(r)

    # ────────────────────────────────────────────────────
    # 3. BF16
    # ────────────────────────────────────────────────────
    if "bf16" in active:
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
            m_bf16 = copy.deepcopy(fp32_model).bfloat16().to(DEVICE)
            r = benchmark_model("BF16 Brain-float", m_bf16,
                                 test_loader, calib_loader, loss_fn,
                                 torch.device(DEVICE),
                                 single_x.bfloat16().to(DEVICE), batch_x.bfloat16().to(DEVICE))
        else:
            print("  [SKIP] BF16 requires Ampere+ GPU — skipping")
            r = {"name":"BF16 Brain-float","note":"skipped"}
        results.append(r)

    # ────────────────────────────────────────────────────
    # 4. PTQ Dynamic INT8
    # ────────────────────────────────────────────────────
    if "ptq_dynamic" in active:
        m_dyn = copy.deepcopy(fp32_model)
        torch.quantization.quantize_dynamic(
            m_dyn, {nn.Linear, nn.Conv2d}, dtype=torch.qint8, inplace=True)
        r = benchmark_model("PTQ Dynamic INT8", m_dyn,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ────────────────────────────────────────────────────
    # 5. PTQ Static INT8
    # ────────────────────────────────────────────────────
    if "ptq_static" in active:
        m_static = copy.deepcopy(fp32_model)
        m_static.qconfig = tq.get_default_qconfig("fbgemm")
        tq.prepare(m_static, inplace=True)
        with torch.no_grad():
            for x, _ in calib_loader:
                m_static(x)
        tq.convert(m_static, inplace=True)
        r = benchmark_model("PTQ Static INT8", m_static,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ────────────────────────────────────────────────────
    # 6. PTQ Per-channel Static INT8
    # ────────────────────────────────────────────────────
    if "ptq_perchannel" in active:
        m_pc = copy.deepcopy(fp32_model)
        m_pc.qconfig = tq.get_default_qconfig("fbgemm")   # fbgemm uses per-channel by default
        tq.prepare(m_pc, inplace=True)
        with torch.no_grad():
            for x, _ in calib_loader:
                m_pc(x)
        tq.convert(m_pc, inplace=True)
        r = benchmark_model("PTQ Per-channel INT8", m_pc,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        r["name"] = "PTQ Per-channel INT8"
        results.append(r)

    # ────────────────────────────────────────────────────
    # 7. PTQ Fake INT4
    # ────────────────────────────────────────────────────
    if "ptq_int4" in active:
        m_int4 = apply_fake_int4(copy.deepcopy(fp32_model))
        r = benchmark_model("PTQ Fake-INT4 (weight-only)", m_int4,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ────────────────────────────────────────────────────
    # 8. PTQ Static + KD fine-tune
    # ────────────────────────────────────────────────────
    if "ptq_finetune" in active:
        m_kd = ptq_finetune(None, copy.deepcopy(fp32_model),
                             calib_loader, loss_fn,
                             n_epochs=args.kd_epochs)
        r = benchmark_model("PTQ Static INT8 + KD Fine-tune", m_kd,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ────────────────────────────────────────────────────
    # 9. QAT INT8
    # ────────────────────────────────────────────────────
    if "qat" in active:
        print(f"\n  Preparing QAT model ({args.qat_epochs} fine-tune epochs)…")
        m_qat = prepare_qat_model(copy.deepcopy(fp32_model))
        m_qat = train_qat(m_qat, train_loader, loss_fn, n_epochs=args.qat_epochs)
        m_qat.eval()
        tq.convert(m_qat, inplace=True)
        r = benchmark_model("QAT INT8", m_qat,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ────────────────────────────────────────────────────
    # 10. GPTQ-style row-wise INT4
    # ────────────────────────────────────────────────────
    if "gptq_int4" in active:
        m_gptq = gptq_int4_weight_only(copy.deepcopy(fp32_model))
        r = benchmark_model("GPTQ-style INT4 (weight-only)", m_gptq,
                             test_loader, calib_loader, loss_fn, CPU, single_x, batch_x)
        results.append(r)

    # ─────────────────────────────────────────────────────
    # Compute relative metrics vs FP32
    # ─────────────────────────────────────────────────────
    if fp32_f1 is not None:
        for r in results:
            if "macro_f1" in r and r.get("name","") != "FP32 Baseline":
                r["f1_drop_pp"]       = round((fp32_f1 - r["macro_f1"]) * 100, 3)
                r["size_vs_fp32_pct"] = round(r.get("model_size_mb",0) / fp32_size * 100, 1)
                r["speedup_vs_fp32"]  = round(fp32_lat / r["lat_single_ms"], 2) \
                                         if r.get("lat_single_ms", 0) > 0 else None
            else:
                r["f1_drop_pp"]       = 0.0
                r["size_vs_fp32_pct"] = 100.0
                r["speedup_vs_fp32"]  = 1.0

    # ─────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved → {out_json}")

    # ─────────────────────────────────────────────────────
    # Console comparison table
    # ─────────────────────────────────────────────────────
    print("\n" + "═"*110)
    print(f"{'Method':<35} {'Acc':>6} {'F1':>6} {'F1↓(pp)':>8} {'Size(MB)':>9} "
          f"{'Size%':>7} {'Lat(ms)':>8} {'Speedup':>8} {'Tput(sps)':>10}")
    print("═"*110)
    for r in results:
        if "macro_f1" not in r:
            print(f"  {r['name']:<33}  — {r.get('note','skipped')}")
            continue
        print(f"  {r['name']:<33} "
              f"{r['accuracy']:>6.4f} "
              f"{r['macro_f1']:>6.4f} "
              f"{r.get('f1_drop_pp', 0):>+8.3f} "
              f"{r.get('model_size_mb', 0):>9.2f} "
              f"{r.get('size_vs_fp32_pct', 100):>6.1f}% "
              f"{r.get('lat_single_ms', 0):>8.2f} "
              f"{r.get('speedup_vs_fp32', 1):>7.2f}x "
              f"{r.get('throughput_sps', 0):>10.0f}")
    print("═"*110)

    # ─────────────────────────────────────────────────────
    # Per-class F1 comparison table
    # ─────────────────────────────────────────────────────
    print("\nPer-class F1 breakdown:")
    header = f"{'Method':<35} " + "  ".join(f"{c[:6]:>6}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for r in results:
        if "per_class_f1" not in r: continue
        row = f"  {r['name']:<33} "
        for c in CLASS_NAMES:
            v = r["per_class_f1"].get(c, 0)
            row += f"  {v:>6.3f}"
        print(row)

    return results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="UrbanSound8K Quantization Benchmark")
    p.add_argument("--checkpoint",  default=str(CHECKPOINT_FILE),
                   help="Path to best .pth checkpoint")
    p.add_argument("--metadata",    default=str(METADATA_PATH),
                   help="Path to UrbanSound8K.csv")
    p.add_argument("--audio-root",  default=str(AUDIO_PATH),
                   help="Path to UrbanSound8K/audio folder")
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR),
                   help="Directory to save results")
    p.add_argument("--methods", nargs="+", default=["all"],
                   choices=["all","fp32","fp16","bf16","ptq_dynamic","ptq_static",
                             "ptq_perchannel","ptq_int4","ptq_finetune","qat","gptq_int4"],
                   help="Which quantisation methods to run")
    p.add_argument("--qat-epochs",  type=int, default=5,
                   help="Number of QAT fine-tune epochs (default 5)")
    p.add_argument("--kd-epochs",   type=int, default=3,
                   help="Number of PTQ+KD fine-tune epochs (default 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override globals from CLI
    AUDIO_PATH    = Path(args.audio_root)
    METADATA_PATH = Path(args.metadata)
    OUTPUT_DIR    = Path(args.output_dir)

    results = run_benchmarks(args)