------

# APA-Core Colab Notebook v1 — Final Locked Implementation Plan

## 0. Document Status

All design decisions are locked. This document is the single source of truth for implementing the notebook. No further architectural debates — only implementation.

------

## 1. Locked Decisions Summary

| Decision             | Choice                                                       | Rationale                                |
| -------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Architecture         | Notebook-first, section-modular, code-inline                 | Colab research workflow                  |
| Config               | Single `Config` frozen dataclass                             | Ergonomic, no multi-object passing       |
| Mode system          | `ModeFlags` dataclass + `MODE_REGISTRY` dict                 | Clean, extensible                        |
| Model abstraction    | Single `ModelSpec` ABC                                       | Minimal but sufficient                   |
| Data flow            | Tuples and dicts, no custom dataclass wrappers               | Practical, debug-friendly                |
| Training core        | `run_single()` with helper decomposition                     | Readable, testable                       |
| Result schema        | Lean dict with optional companion `.npz`                     | JSON stays small, DVA/LLM data available |
| Normalization        | `train_channel_zscore` (fit train, transform all)            | Matches ATCNet original preprocessing    |
| Preprocessing order  | load → split → optional EA → zscore                          | Fair comparison, no leakage              |
| Augmentation         | On-the-fly, batch-level, torch tensors, train only           | Standard DL practice                     |
| Validation           | Every epoch, 48-trial val set                                | Low cost, full checkpoint coverage       |
| MI window            | [2.0s, 6.0s] = 1000 samples                                  | Unified cross-model comparison           |
| Benchmark files      | Smoke and full write to separate JSONs                       | No cross-contamination                   |
| Resume validation    | Strict field-by-field, error on mismatch                     | Prevents silent data corruption          |
| Resume exempt fields | `output_dir`, `fail_fast`, `seed`, `save_training_history`, `save_trial_outputs` | Don’t affect experiment identity         |
| Wilcoxon             | `zero_method="wilcox"`, effective N, one-sided primary       | Statistically correct                    |
| ML support           | Not implemented in v1, future hooks via `ModelSpec.family`   | DL-first, ML-ready                       |
| DVA/LLM              | Not implemented, `y_proba` saved as hook                     | Cheap future-proofing                    |
| Plotting             | Inline in notebook                                           | Simplest for research workflow           |

------

## 2. Section Map

```
Section 1:  Introduction & Scope                 [1 markdown cell]
Section 2:  Environment Setup                     [2 code cells]
Section 3:  Configuration & Registries            [1 code cell]
Section 4:  Data Loader                           [1 code cell, ~250 lines]
Section 5:  Data Verification & Exploration       [3 code cells with plots]
Section 6:  APA-Core Components                   [1 code cell, ~300 lines]
Section 7:  Model Specs & Registry                [1 code cell, ~80 lines]
Section 8:  Orchestrator (run_single)             [1 code cell, ~250 lines]
Section 9:  Benchmark Runner                      [1 code cell, ~250 lines]
Section 10: Smoke Benchmark                       [2 code cells]
Section 11: Full Benchmark                        [2 code cells]
Section 12: Results & Visualization               [6-7 code cells]
Section 13: Statistical Analysis                  [2 code cells]
Section 14: Discussion, Export & Future Hooks      [2 cells: 1 markdown, 1 code]
```

Total: ~14 sections, ~25 cells, ~1800 lines of code.

------

## 3. Section-by-Section Specification

------

### Section 1 — Introduction & Scope

**Type:** Markdown only
 **Purpose:** Frame the research question and notebook scope

**Content outline:**

```markdown
# APA-Core: Lightweight Model-Agnostic Enhancements for MI-BCI

## What is APA-Core?
Four simple, model-agnostic components added to published DL classifiers:
1. Euclidean Alignment (EA) — session-wise covariance whitening
2. Mild Data Augmentation — Gaussian noise + amplitude scaling + temporal jitter
3. Cosine Annealing LR — smooth learning rate decay
4. Top-k Checkpoint Averaging — ensemble of best k validation checkpoints

## This Notebook
- Model: ATCNet (via braindecode)
- Dataset: BCI Competition IV-2a (9 subjects, 4-class MI, 22 EEG, 250 Hz)
- Evaluation: Session-wise (train on T, test on E)
- Comparison: controlled_baseline vs apacore, 5 seeds per subject
- Statistical test: One-sided Wilcoxon signed-rank on per-subject mean accuracies

## Important Note on MI Window
This notebook uses a unified [2s, 6s] = 1000-sample window for fair cross-model
comparison. ATCNet's original paper uses [1.5s, 6s] = 1125 samples. Published
ATCNet results should be compared with this difference in mind.

## V1 Scope
- DL-first (ATCNet). Other DL models added in later versions.
- No ML classifiers, DVA, or LLM explainability in v1.
- Trial-level outputs (y_proba) saved for future DVA/LLM integration.
```

------

### Section 2 — Environment Setup

**Cell 2a — Dependencies & Paths:**

```python
# ==============================================================================
# Section 2a: Environment Setup
# ==============================================================================
import os, sys, subprocess, warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules
print(f"Running on Google Colab: {IN_COLAB}")

# ── Dependencies ──
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn',
    'pandas', 'braindecode', 'einops'], capture_output=True)

import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("  ⚠ No GPU detected. Training will be very slow.")

# ── Paths ──
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_DIR = '/content/drive/MyDrive/LLM-EEG'
    DATA_DIR = os.path.join(DRIVE_DIR, 'data')
    RESULTS_DIR = os.path.join(DRIVE_DIR, 'results', 'apa_core')
else:
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data Check (NO synthetic fallback) ──
missing = []
for subj in range(1, 10):
    for sess in ['T', 'E']:
        path = os.path.join(DATA_DIR, f'A0{subj}{sess}.mat')
        if not os.path.exists(path):
            missing.append(f'A0{subj}{sess}.mat')

if missing:
    print(f"\n✗ Missing {len(missing)} file(s) in {DATA_DIR}:")
    for f in missing[:10]:
        print(f"    {f}")
    print(f"\n  Download from: https://bnci-horizon-2020.eu/database/data-sets")
    print(f"  (BCI Competition IV-2a, dataset 001-2014)")
    if any('E.mat' in f for f in missing) and not any('T.mat' in f for f in missing):
        print(f"\n  ⚠ Training files found but evaluation files missing.")
        print(f"    Eval labels may need sidecar .txt files.")
        print(f"    Download from: https://www.bbci.de/competition/iv/results/")
else:
    print(f"\n✓ All 18 data files found in {DATA_DIR}")

print(f"\nData:    {DATA_DIR}")
print(f"Results: {RESULTS_DIR}")
```

**Cell 2b — Imports & Global Constants:**

```python
# ==============================================================================
# Section 2b: Imports & Constants
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy.io as sio
import scipy.signal as scipy_signal
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Optional, Literal, Any, Dict, List, Tuple
from datetime import datetime, timezone
import json, time, copy, random, tempfile, traceback, inspect, platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Plotting Style ──
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'figure.figsize': (12, 6), 'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.titlesize': 15, 'lines.linewidth': 1.8,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
COLORS = {
    'controlled_baseline': '#4C72B0',
    'apacore': '#DD8452',
    'ablate': '#55A868',
}

# ── Global Constants ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUBJECTS = list(range(1, 10))
SEEDS = [42, 123, 456, 789, 1024]
N_REQUIRED_SEEDS = len(SEEDS)

CHANNEL_NAMES_22 = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz',
]
N_CLASSES = 4
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
FS = 250

print(f"Device: {DEVICE}")
print(f"Subjects: {SUBJECTS}")
print(f"Seeds: {SEEDS}")
```

------

### Section 3 — Configuration & Registries

```python
# ==============================================================================
# Section 3: Configuration & Registries
# ==============================================================================

# ── Config ──
@dataclass(frozen=True)
class Config:
    # Paths
    data_dir: str = "data/"
    output_dir: str = "results/"
    eval_labels_path: Optional[str] = None  # supports "{subject}" template

    # Data
    drop_artifacts: bool = True
    t_start: float = 2.0      # MI window start (seconds post trial onset)
    t_end: float = 6.0        # MI window end
    normalize: Literal["none", "train_channel_zscore"] = "train_channel_zscore"

    # Split
    val_run_id: int = 6        # hold out this run for validation
    stratified_val_frac: float = 0.2  # fallback if run_ids unavailable

    # EA
    ea_shrinkage: float = 0.01
    ea_eig_floor: float = 1e-6

    # Augmentation
    aug_noise_std_frac: float = 0.05
    aug_scale_range: tuple = (0.9, 1.1)
    aug_jitter_max: int = 10

    # LR / Checkpoint
    lr_eta_min: float = 1e-6
    topk: int = 3
    min_epoch_gap: int = 5
    checkpoint_metric: Literal["accuracy", "kappa"] = "accuracy"

    # Benchmark
    seed: int = 42
    fail_fast: bool = False

    # Optional outputs
    save_trial_outputs: bool = False
    save_training_history: bool = False

    @property
    def n_times(self):
        return int((self.t_end - self.t_start) * FS)

    @property
    def mi_start_sample(self):
        return int(self.t_start * FS)

    @property
    def mi_end_sample(self):
        return int(self.t_end * FS)


# ── Mode Flags ──
@dataclass(frozen=True)
class ModeFlags:
    use_ea: bool
    use_aug: bool
    use_cosine: bool
    use_topk: bool

MODE_REGISTRY = {
    'controlled_baseline': ModeFlags(False, False, False, False),
    'apacore':             ModeFlags(True,  True,  True,  True),
    'ablate_no_ea':        ModeFlags(False, True,  True,  True),
    'ablate_no_aug':       ModeFlags(True,  False, True,  True),
    'ablate_no_cosine':    ModeFlags(True,  True,  False, True),
    'ablate_no_topk':      ModeFlags(True,  True,  True,  False),
}

# ── Resume Exempt Fields ──
_RESUME_EXEMPT_FIELDS = {"output_dir", "fail_fast", "seed",
                          "save_training_history", "save_trial_outputs"}

# ── Create configs for this notebook ──
CONFIG = Config(data_dir=DATA_DIR, output_dir=RESULTS_DIR)

print(f"Config: n_times={CONFIG.n_times}, MI window=[{CONFIG.t_start}s, {CONFIG.t_end}s]")
print(f"Modes available: {list(MODE_REGISTRY.keys())}")
```

------

### Section 4 — Data Loader

```python
# ==============================================================================
# Section 4: Data Loader
# ==============================================================================
# Future module: apacore/data_loader.py

_VALID_LABELS = {1, 2, 3, 4}

def _safe_int_set(arr):
    """Convert array values to int set. Returns None if NaN/Inf present."""
    result = set()
    for v in arr:
        try:
            if np.issubdtype(type(v), np.floating) and (np.isnan(v) or np.isinf(v)):
                return None
            result.add(int(v))
        except (ValueError, OverflowError):
            return None
    return result

def _labels_available(a_y_raw, n_trials):
    """Check if label array contains valid class labels (NaN-safe)."""
    if a_y_raw.size < n_trials:
        return False
    trial_labels = _safe_int_set(a_y_raw[:n_trials])
    if trial_labels is None:
        return False
    return trial_labels.issubset(_VALID_LABELS) and len(trial_labels) > 0

def _describe_labels(a_y_raw, n_trials):
    """Safely describe label values for error messages."""
    chunk = a_y_raw[:min(a_y_raw.size, n_trials)]
    safe = _safe_int_set(chunk)
    if safe is not None:
        return str(safe)
    unique = np.unique(chunk)
    if len(unique) <= 10:
        return f"raw unique values: {unique.tolist()}"
    return f"raw unique values (first 10): {unique[:10].tolist()}..."

def _resolve_sidecar_labels(subject_id, session, config, expected_n=None):
    """Load sidecar label file with validation."""
    if config.eval_labels_path is None:
        return None
    path = config.eval_labels_path
    if '{subject' in path:
        path = path.format(subject=subject_id)
    if not os.path.exists(path):
        return None
    labels = np.loadtxt(path, dtype=int)
    if expected_n is not None:
        if labels.size < expected_n:
            raise ValueError(
                f"Sidecar '{path}' has {labels.size} labels but {expected_n} required.")
        if labels.size > expected_n:
            if labels.size / expected_n > 1.5:
                raise ValueError(
                    f"Sidecar '{path}' has {labels.size} labels but only {expected_n} "
                    f"expected (ratio {labels.size/expected_n:.1f}x). Wrong file?")
            labels = labels[:expected_n]
    return labels

def load_bci2a_session(subject_id, session, config):
    """Load one session of BCI IV-2a data.

    Returns:
        X:        (n_trials, 22, n_times) float32
        y:        (n_trials,) int64, 0-indexed
        run_ids:  (n_trials,) int64, 1-indexed (None for flat fallback)
        raw_runs: list of (signal, onsets) per run
    """
    fname = f"A{subject_id:02d}{session}.mat"
    path = os.path.join(config.data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    mat = sio.loadmat(path, squeeze_me=False)
    a_data = mat['data']

    n_channels = 22
    mi_start = config.mi_start_sample
    mi_end = config.mi_end_sample
    n_times = config.n_times

    all_X, all_y, all_run_ids = [], [], []
    raw_runs = []
    run_counter = 0

    # Count total expected trials for sidecar validation
    total_expected = 0
    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = a_data1[0, 0]
        a_trial = np.array(a_data2[1]).flatten()
        if a_trial.size > 0:
            total_expected += a_trial.size

    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = a_data1[0, 0]
        a_X = np.array(a_data2[0], dtype=np.float64)
        a_trial = np.array(a_data2[1]).flatten()
        a_y_raw = np.array(a_data2[2]).flatten()
        a_artifacts = (np.array(a_data2[5]).flatten()
                       if a_data2[5].size > 0 else np.zeros(0))

        if a_trial.size == 0:
            continue

        run_counter += 1
        raw_runs.append((a_X, a_trial))

        has_labels = _labels_available(a_y_raw, a_trial.size)
        sidecar_labels = None
        if not has_labels:
            sidecar_labels = _resolve_sidecar_labels(
                subject_id, session, config, expected_n=total_expected)
            if sidecar_labels is None:
                raise FileNotFoundError(
                    f"Labels missing for S{subject_id}{session} run {run_counter}. "
                    f"Found {_describe_labels(a_y_raw, a_trial.size)}, "
                    f"expected subset of {_VALID_LABELS}.\n"
                    f"Provide eval_labels_path or download from "
                    f"https://www.bbci.de/competition/iv/results/")

        for t_idx in range(a_trial.size):
            if (config.drop_artifacts and t_idx < a_artifacts.size
                    and a_artifacts[t_idx] != 0):
                continue

            onset = int(a_trial[t_idx])
            start = onset + mi_start
            end = onset + mi_end
            if end > a_X.shape[0]:
                continue

            trial = a_X[start:end, :n_channels].T  # (22, n_times)

            if has_labels:
                label = int(a_y_raw[t_idx]) - 1
            else:
                global_idx = sum(len(r[1]) for r in raw_runs[:-1]) + t_idx
                label = int(sidecar_labels[global_idx]) - 1

            all_X.append(trial)
            all_y.append(label)
            all_run_ids.append(run_counter)

    if len(all_X) == 0:
        raise RuntimeError(f"No valid trials for S{subject_id}{session}")

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    run_ids = np.array(all_run_ids, dtype=np.int64)

    return X, y, run_ids, raw_runs

# ── Quick loader test ──
try:
    _X, _y, _r, _ = load_bci2a_session(1, 'T', CONFIG)
    print(f"✓ Loader test: S01T → X={_X.shape}, y={_y.shape}, "
          f"runs={np.unique(_r)}, classes={np.unique(_y)}")
    del _X, _y, _r
except Exception as e:
    print(f"✗ Loader test failed: {e}")
```

------

### Section 5 — Data Verification & Exploration

**Cell 5a — Shape & Distribution Checks:**

```python
# ==============================================================================
# Section 5a: Data Verification
# ==============================================================================
X_demo, y_demo, run_ids_demo, raw_runs_demo = load_bci2a_session(1, 'T', CONFIG)

print(f"Subject 1 Training Session:")
print(f"  X shape:       {X_demo.shape}  (trials, channels, samples)")
print(f"  y shape:       {y_demo.shape}")
print(f"  Classes:       {np.unique(y_demo)} → counts={[int(np.sum(y_demo==c)) for c in range(4)]}")
print(f"  Runs:          {np.unique(run_ids_demo)} → counts={[int(np.sum(run_ids_demo==r)) for r in np.unique(run_ids_demo)]}")
print(f"  MI window:     [{CONFIG.t_start}s, {CONFIG.t_end}s] = {CONFIG.n_times} samples")
print(f"  Channels:      {len(CHANNEL_NAMES_22)}")
print(f"  Sampling rate: {FS} Hz")

# Verify expected counts
assert X_demo.shape[1] == 22, f"Expected 22 channels, got {X_demo.shape[1]}"
assert X_demo.shape[2] == CONFIG.n_times, f"Expected {CONFIG.n_times} samples, got {X_demo.shape[2]}"
assert set(y_demo.tolist()).issubset({0,1,2,3}), f"Unexpected labels: {np.unique(y_demo)}"
print("\n✓ All shape/label checks passed")
```

**Cell 5b — Onset Verification Plot:**

```python
# ==============================================================================
# Section 5b: Onset Verification
# ==============================================================================
if raw_runs_demo:
    signal, onsets = raw_runs_demo[0]
    n_plot = min(3, len(onsets))
    ch_idx = 9  # Cz

    fig, axes = plt.subplots(n_plot, 1, figsize=(16, 3 * n_plot))
    if n_plot == 1: axes = [axes]

    for i in range(n_plot):
        onset = int(onsets[i])
        ps = max(0, onset - FS)
        pe = min(signal.shape[0], onset + 7 * FS)
        t = np.arange(ps, pe) / FS

        axes[i].plot(t, signal[ps:pe, ch_idx], lw=0.5, color='steelblue')
        axes[i].axvline(x=onset/FS, color='green', lw=2, label='Onset')
        ws = (onset + CONFIG.mi_start_sample) / FS
        we = (onset + CONFIG.mi_end_sample) / FS
        axes[i].axvline(x=ws, color='red', ls='--', lw=1.5, label=f'+{CONFIG.t_start}s')
        axes[i].axvline(x=we, color='red', ls='--', lw=1.5, label=f'+{CONFIG.t_end}s')
        axes[i].axvspan(ws, we, alpha=0.12, color='red')
        axes[i].set_ylabel(f'Trial {i}')
        if i == 0: axes[i].legend(fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'S01 Run 1 — {CHANNEL_NAMES_22[ch_idx]} — MI Window Verification', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'verify_onsets_S01.png'))
    plt.show()
    print("✓ Verify green=onset, red shading=MI extraction window")
```

**Cell 5c — Raw EEG & PSD:**

```python
# ==============================================================================
# Section 5c: Raw EEG & PSD Exploration
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (a) Raw EEG
trial = X_demo[0]
t = np.arange(trial.shape[1]) / FS
for i, (name, idx) in enumerate([('C3',7),('C4',11),('Cz',9),('Fz',0)]):
    axes[0,0].plot(t, trial[idx] + i*50, label=name, linewidth=0.8)
axes[0,0].set_xlabel('Time (s)'); axes[0,0].set_ylabel('Amplitude')
axes[0,0].set_title('(a) Raw EEG Trial 1'); axes[0,0].legend()

# (b) PSD by class at C3
for ci, cn in enumerate(CLASS_NAMES):
    ct = X_demo[y_demo == ci]
    if len(ct) == 0: continue
    psds = [scipy_signal.welch(tr[7], fs=FS, nperseg=min(256, tr.shape[1]))[1]
            for tr in ct[:10]]
    f_ax = scipy_signal.welch(ct[0,7], fs=FS, nperseg=min(256, ct[0].shape[1]))[0]
    axes[0,1].semilogy(f_ax, np.mean(psds, axis=0), label=cn, linewidth=1.5)
axes[0,1].set_xlim(0, 50); axes[0,1].set_xlabel('Frequency (Hz)')
axes[0,1].set_ylabel('PSD'); axes[0,1].set_title('(b) PSD at C3 by Class')
axes[0,1].legend(); axes[0,1].axvspan(8,12,alpha=0.15,color='red')
axes[0,1].axvspan(12,30,alpha=0.1,color='blue')

# (c) Class distribution
counts = [int(np.sum(y_demo==c)) for c in range(4)]
axes[1,0].bar(CLASS_NAMES, counts, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
axes[1,0].set_ylabel('Count'); axes[1,0].set_title('(c) Class Distribution')

# (d) Channel variance
ch_vars = np.var(trial, axis=1)
axes[1,1].barh(CHANNEL_NAMES_22, ch_vars, color='#55A868', alpha=0.8)
axes[1,1].set_xlabel('Variance'); axes[1,1].set_title('(d) Channel Variance (Trial 1)')
axes[1,1].invert_yaxis()

plt.suptitle('Data Exploration — Subject 1 Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'data_exploration.png'))
plt.show()

del X_demo, y_demo, run_ids_demo, raw_runs_demo
```

------

### Section 6 — APA-Core Components

```python
# ==============================================================================
# Section 6: APA-Core Components
# ==============================================================================
# Future module: apacore/components.py

# ── 6.1 Split ──
def smart_split(X, y, run_ids, config):
    """Run-based validation split. Fallback: stratified."""
    if run_ids is not None and len(np.unique(run_ids)) > 1:
        mask_val = (run_ids == config.val_run_id)
        if mask_val.sum() > 0:
            return X[~mask_val], y[~mask_val], X[mask_val], y[mask_val]
    # Fallback
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.stratified_val_frac,
                                  random_state=config.seed)
    tr_idx, va_idx = next(sss.split(X, y))
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


# ── 6.2 Euclidean Alignment ──
class DiagonalShrinkageEA:
    """Fit on training data only. Transform all splits with same W."""
    def __init__(self, shrinkage=0.01, eig_floor=1e-6):
        self.shrinkage = shrinkage
        self.eig_floor = eig_floor
        self.W_ = None

    def fit(self, X):
        n, c, t = X.shape
        C = np.zeros((c, c), dtype=np.float64)
        for i in range(n):
            C += X[i] @ X[i].T / t
        C /= n
        C_reg = (1 - self.shrinkage) * C + self.shrinkage * np.eye(c)
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        eigvals = np.maximum(eigvals, self.eig_floor)
        self.W_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return self

    def transform(self, X):
        if self.W_ is None:
            raise RuntimeError("Call fit() before transform().")
        return np.stack([self.W_ @ X[i] for i in range(X.shape[0])])


# ── 6.3 Normalization ──
def apply_train_channel_zscore(X_train, X_val, X_test):
    """Fit StandardScaler per channel on X_train, transform all.
    This matches ATCNet's original preprocessing."""
    n_channels = X_train.shape[1]
    X_tr = X_train.copy()
    X_va = X_val.copy() if X_val is not None else None
    X_te = X_test.copy()

    for ch in range(n_channels):
        scaler = StandardScaler()
        # Fit on train: (n_train_trials, n_times)
        scaler.fit(X_tr[:, ch, :])
        X_tr[:, ch, :] = scaler.transform(X_tr[:, ch, :])
        if X_va is not None:
            X_va[:, ch, :] = scaler.transform(X_va[:, ch, :])
        X_te[:, ch, :] = scaler.transform(X_te[:, ch, :])

    return X_tr, X_va, X_te


# ── 6.4 Mild Augmenter (torch, batch-level) ──
class MildAugmenter:
    """Applied on-the-fly during training, on GPU tensors."""
    def __init__(self, noise_std_frac=0.05, scale_range=(0.9, 1.1), jitter_max=10):
        self.noise_std_frac = noise_std_frac
        self.scale_range = scale_range
        self.jitter_max = jitter_max

    def __call__(self, X):
        """X: (B, C, T) torch tensor on device."""
        std = X.std(dim=-1, keepdim=True)
        noise = torch.randn_like(X) * std * self.noise_std_frac
        scale = torch.empty(X.shape[0], 1, 1, device=X.device).uniform_(*self.scale_range)
        shift = torch.randint(-self.jitter_max, self.jitter_max + 1, (X.shape[0],))

        X_aug = X * scale + noise
        X_out = torch.zeros_like(X_aug)
        for i in range(X.shape[0]):
            s = shift[i].item()
            if s > 0:   X_out[i, :, s:] = X_aug[i, :, :-s]
            elif s < 0:  X_out[i, :, :s] = X_aug[i, :, -s:]
            else:        X_out[i] = X_aug[i]
        return X_out


# ── 6.5 Checkpoint Manager ──
class _CkptEntry:
    def __init__(self, epoch, val_metric, state_dict):
        self.epoch = epoch
        self.val_metric = val_metric
        self.state_dict = state_dict

class CheckpointManager:
    """Top-k checkpoints with min epoch gap. CPU storage."""
    def __init__(self, topk=3, min_epoch_gap=5):
        self.topk = topk
        self.min_epoch_gap = min_epoch_gap
        self.entries = []

    def update(self, epoch, val_metric, model):
        # Reject if within gap of a better entry
        for e in self.entries:
            if abs(epoch - e.epoch) < self.min_epoch_gap and e.val_metric >= val_metric:
                return
        # Remove worse entries within gap
        self.entries = [e for e in self.entries
                        if not (abs(epoch - e.epoch) < self.min_epoch_gap
                                and val_metric > e.val_metric)]
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.entries.append(_CkptEntry(epoch, val_metric, sd))
        self.entries.sort(key=lambda e: e.val_metric, reverse=True)
        self.entries = self.entries[:self.topk]

    def best_state_dict(self):
        if not self.entries:
            raise RuntimeError("No checkpoints saved.")
        return self.entries[0].state_dict

    def average_predictions(self, model, X_tensor, device, batch_size,
                             reshape_fn=None, extract_fn=None):
        all_probs = []
        for entry in self.entries:
            model.load_state_dict(entry.state_dict)
            model.to(device)
            all_probs.append(
                _batched_predict(model, X_tensor, device, batch_size,
                                 reshape_fn, extract_fn))
        return np.mean(all_probs, axis=0)


def _batched_predict(model, X, device, batch_size=64,
                      reshape_fn=None, extract_fn=None):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start+batch_size].to(device)
            if reshape_fn: xb = reshape_fn(xb)
            out = model(xb)
            logits = extract_fn(out) if extract_fn else out
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ── 6.6 Metrics ──
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def cohen_kappa(y_true, y_pred, n_classes=4):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    n = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / n**2
    return float((po - pe) / (1 - pe + 1e-10))

def confusion_matrix_4class(y_true, y_pred):
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


print("✓ Components loaded: split, EA, zscore, augmenter, checkpoint, metrics")
```

------

### Section 7 — Model Specs & Registry

```python
# ==============================================================================
# Section 7: Model Specs & Registry
# ==============================================================================
# Future module: models/base.py, models/atcnet.py, models/registry.py

class ModelSpec(ABC):
    """Base class for DL model specifications."""
    name: str = ""
    family: str = "torch"          # future: "sklearn"
    feature_pipeline: str = "raw"  # future: "csp", "fbcsp"

    @abstractmethod
    def build_model(self, n_classes: int, n_channels: int, n_times: int) -> nn.Module:
        ...

    @abstractmethod
    def default_train_params(self) -> dict:
        """Return {'epochs': ..., 'lr': ..., 'batch_size': ..., 'weight_decay': ...}"""
        ...

    def reshape_input(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def extract_logits(self, model_output) -> torch.Tensor:
        return model_output


class ATCNetSpec(ModelSpec):
    name = "ATCNet"

    def build_model(self, n_classes=4, n_channels=22, n_times=1000):
        from braindecode.models import ATCNet
        sig = inspect.signature(ATCNet)
        if 'n_times' in sig.parameters:
            return ATCNet(n_chans=n_channels, n_outputs=n_classes, n_times=n_times)
        elif 'input_window_seconds' in sig.parameters:
            return ATCNet(n_chans=n_channels, n_outputs=n_classes,
                          input_window_seconds=n_times / FS, sfreq=FS)
        else:
            raise ImportError("Cannot determine ATCNet constructor signature.")

    def default_train_params(self):
        return {
            'epochs': 500,
            'lr': 9e-4,
            'batch_size': 64,
            'weight_decay': 0.0,
        }


# ── Registry ──
MODEL_REGISTRY = {
    'atcnet': ATCNetSpec(),
    # Future:
    # 'eegconformer': EEGConformerSpec(),
    # 'mscformer': MSCFormerSpec(),
    # 'cltnet': CLTNetSpec(),
}

# ── Verify ATCNet builds ──
try:
    _spec = MODEL_REGISTRY['atcnet']
    _model = _spec.build_model(n_classes=4, n_channels=22, n_times=CONFIG.n_times)
    _n_params = sum(p.numel() for p in _model.parameters())
    _dummy = torch.randn(2, 22, CONFIG.n_times)
    _out = _model(_spec.reshape_input(_dummy))
    _logits = _spec.extract_logits(_out)
    assert _logits.shape == (2, 4), f"Expected (2,4), got {_logits.shape}"
    print(f"✓ ATCNet: {_n_params:,} params, output shape {_logits.shape}")
    del _model, _dummy, _out, _logits
except Exception as e:
    print(f"✗ ATCNet build failed: {e}")
```

------

### Section 8 — Orchestrator (`run_single`)

```python
# ==============================================================================
# Section 8: Orchestrator
# ==============================================================================
# Future module: apacore/orchestrator.py

def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _prepare_data(subject_id, mode_flags, config):
    """Load T+E sessions, split, apply EA + zscore.

    Returns: X_train, y_train, X_val, y_val, X_test, y_test (all np arrays)
    """
    X_T, y_T, run_ids, _ = load_bci2a_session(subject_id, 'T', config)
    X_E, y_E, _, _ = load_bci2a_session(subject_id, 'E', config)

    X_train, y_train, X_val, y_val = smart_split(X_T, y_T, run_ids, config)
    X_test, y_test = X_E, y_E

    # Optional EA (fit on train only)
    if mode_flags.use_ea:
        ea = DiagonalShrinkageEA(config.ea_shrinkage, config.ea_eig_floor)
        ea.fit(X_train)
        X_train = ea.transform(X_train)
        X_val = ea.transform(X_val)
        X_test = ea.transform(X_test)

    # Normalization
    if config.normalize == "train_channel_zscore":
        X_train, X_val, X_test = apply_train_channel_zscore(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_model(spec, X_train, y_train, X_val, y_val,
                 mode_flags, config, device):
    """Build model, train, return model + checkpoint manager + timing info."""
    params = spec.default_train_params()
    n_channels, n_times = X_train.shape[1], X_train.shape[2]

    model = spec.build_model(n_classes=N_CLASSES, n_channels=n_channels,
                              n_times=n_times).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                            weight_decay=params['weight_decay'])
    scheduler = (CosineAnnealingLR(optimizer, T_max=params['epochs'],
                                    eta_min=config.lr_eta_min)
                 if mode_flags.use_cosine else None)
    criterion = nn.CrossEntropyLoss()

    augmenter = (MildAugmenter(config.aug_noise_std_frac,
                                config.aug_scale_range,
                                config.aug_jitter_max)
                 if mode_flags.use_aug else None)
    ckpt = CheckpointManager(config.topk, config.min_epoch_gap)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    g = torch.Generator(); g.manual_seed(config.seed)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                         batch_size=params['batch_size'],
                         shuffle=True, generator=g,
                         worker_init_fn=_seed_worker)

    train_start = time.time()
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if augmenter is not None:
                xb = augmenter(xb)
            xb = spec.reshape_input(xb)
            logits = spec.extract_logits(model(xb))
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        # Validation (every epoch)
        val_probs = _batched_predict(model, X_val_t, device,
                                      batch_size=params['batch_size'],
                                      reshape_fn=spec.reshape_input,
                                      extract_fn=spec.extract_logits)
        val_preds = np.argmax(val_probs, axis=1)
        val_metric = accuracy(y_val, val_preds)
        ckpt.update(epoch, val_metric, model)

        if config.save_training_history:
            history['train_loss'].append(epoch_loss / max(n_batches, 1))
            history['val_acc'].append(val_metric)

    train_time = time.time() - train_start
    return model, ckpt, train_time, history


def _evaluate_model(spec, model, ckpt, X_test, y_test,
                     mode_flags, config, device):
    """Run inference, compute metrics, return result info."""
    params = spec.default_train_params()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    infer_start = time.time()

    if mode_flags.use_topk and ckpt.entries:
        test_probs = ckpt.average_predictions(
            model, X_test_t, device,
            batch_size=params['batch_size'],
            reshape_fn=spec.reshape_input,
            extract_fn=spec.extract_logits)
    elif ckpt.entries:
        model.load_state_dict(ckpt.best_state_dict())
        model.to(device)
        test_probs = _batched_predict(
            model, X_test_t, device,
            batch_size=params['batch_size'],
            reshape_fn=spec.reshape_input,
            extract_fn=spec.extract_logits)
    else:
        test_probs = _batched_predict(
            model, X_test_t, device,
            batch_size=params['batch_size'],
            reshape_fn=spec.reshape_input,
            extract_fn=spec.extract_logits)

    infer_time = time.time() - infer_start
    test_preds = np.argmax(test_probs, axis=1)

    eval_info = {
        'accuracy': accuracy(y_test, test_preds),
        'kappa': cohen_kappa(y_test, test_preds),
        'confusion_matrix': confusion_matrix_4class(y_test, test_preds),
        'inference_time_sec': round(infer_time, 2),
        'y_true': y_test.tolist(),
        'y_pred': test_preds.tolist(),
        'y_proba': test_probs.tolist(),
    }
    return eval_info


def run_single(spec, subject_id, mode_name, config, device):
    """Full pipeline: prepare → train → evaluate → result dict."""
    mode_flags = MODE_REGISTRY[mode_name]
    _seed_everything(config.seed)

    X_tr, y_tr, X_va, y_va, X_te, y_te = _prepare_data(
        subject_id, mode_flags, config)

    model, ckpt, train_time, history = _train_model(
        spec, X_tr, y_tr, X_va, y_va, mode_flags, config, device)

    eval_info = _evaluate_model(
        spec, model, ckpt, X_te, y_te, mode_flags, config, device)

    result = {
        'model': spec.name,
        'mode': mode_name,
        'subject': subject_id,
        'seed': config.seed,
        'accuracy': eval_info['accuracy'],
        'kappa': eval_info['kappa'],
        'confusion_matrix': eval_info['confusion_matrix'],
        'train_time_sec': round(train_time, 2),
        'inference_time_sec': eval_info['inference_time_sec'],
        'n_train': len(y_tr),
        'n_val': len(y_va),
        'n_test': len(y_te),
        'status': 'ok',
        'error': None,
    }

    # Optional trial outputs (stored separately later)
    trial_outputs = None
    if config.save_trial_outputs:
        trial_outputs = {
            'y_true': eval_info['y_true'],
            'y_pred': eval_info['y_pred'],
            'y_proba': eval_info['y_proba'],
        }

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result, trial_outputs, history


print("✓ Orchestrator loaded: _prepare_data, _train_model, _evaluate_model, run_single")
```

------

### Section 9 — Benchmark Runner

```python
# ==============================================================================
# Section 9: Benchmark Runner
# ==============================================================================
# Future module: apacore/runner.py

def _make_run_key(mode, subject, seed):
    return f"{mode}__S{subject:02d}__seed{seed}"

def _build_completed_set(results_list):
    return {_make_run_key(r['mode'], r['subject'], r['seed'])
            for r in results_list if r.get('status') == 'ok'}

def _generate_output_path(output_dir, model_name, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{model_name}_{tag}_{ts}.json")

def _atomic_json_write(data, path):
    dir_name = os.path.dirname(path) or '.'
    fd = None; tmp_path = None
    try:
        fd = tempfile.NamedTemporaryFile(mode='w', dir=dir_name,
                                          suffix='.tmp', prefix='.apa_',
                                          delete=False)
        tmp_path = fd.name
        json.dump(data, fd, indent=2)
        fd.flush(); os.fsync(fd.fileno()); fd.close(); fd = None
        os.replace(tmp_path, path); tmp_path = None
    finally:
        if fd is not None: fd.close()
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)

def _collect_metadata(model_name, modes, config, subjects, seeds):
    return {
        "plan_version": "notebook_v1",
        "model": model_name,
        "modes": list(modes),
        "subjects": list(subjects),
        "seeds": list(seeds),
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in config.__dict__.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(DEVICE),
    }

def _validate_resume_metadata(saved_meta, model_name, modes, config, subjects, seeds):
    errors = []
    if saved_meta.get("model") != model_name:
        errors.append(f"model: '{saved_meta.get('model')}' vs '{model_name}'")
    if set(saved_meta.get("modes", [])) != set(modes):
        errors.append(f"modes mismatch")
    if set(saved_meta.get("subjects", [])) != set(subjects):
        errors.append(f"subjects mismatch")
    if set(saved_meta.get("seeds", [])) != set(seeds):
        errors.append(f"seeds mismatch")
    saved_cfg = saved_meta.get("config", {})
    for f in fields(config):
        if f.name in _RESUME_EXEMPT_FIELDS:
            continue
        saved_val = saved_cfg.get(f.name)
        current_val = getattr(config, f.name)
        if isinstance(current_val, tuple):
            current_val = list(current_val)
        if saved_val != current_val:
            errors.append(f"config.{f.name}: {saved_val!r} vs {current_val!r}")
    if errors:
        raise ValueError("Resume validation failed:\n  " + "\n  ".join(errors))


def run_full_benchmark(spec, device, config, modes, subjects=None,
                        seeds=None, resume_path=None, tag="full"):
    """Multi-seed, multi-subject, multi-mode benchmark with resume."""
    if subjects is None: subjects = SUBJECTS
    if seeds is None: seeds = SEEDS

    os.makedirs(config.output_dir, exist_ok=True)
    metadata = _collect_metadata(spec.name, modes, config, subjects, seeds)

    # ── Resume logic ──
    all_results = []
    failures = []
    trial_outputs_accum = {}
    completed = set()

    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        with open(resume_path) as f:
            saved = json.load(f)
        if isinstance(saved, dict) and 'results' in saved:
            _validate_resume_metadata(
                saved.get('metadata', {}), spec.name, modes, config, subjects, seeds)
            all_results = saved['results']
            failures = saved.get('failures', [])
        completed = _build_completed_set(all_results)
        out_path = resume_path
        print(f"  Resumed {len(completed)} completed runs from {resume_path}")
    else:
        out_path = _generate_output_path(config.output_dir, spec.name, tag)

    total = len(modes) * len(subjects) * len(seeds)
    done_count = len(completed)

    for mode in modes:
        for subj in subjects:
            for seed in seeds:
                key = _make_run_key(mode, subj, seed)
                if key in completed:
                    continue

                run_config = replace(config, seed=seed)

                try:
                    result, trial_out, history = run_single(
                        spec, subj, mode, run_config, device)
                    all_results.append(result)
                    done_count += 1

                    # Remove stale failure for this run
                    failures = [f for f in failures
                                if _make_run_key(f['mode'], f['subject'], f['seed']) != key]

                    if trial_out:
                        prefix = f"S{subj:02d}_{mode}_seed{seed}"
                        for k, v in trial_out.items():
                            trial_outputs_accum[f"{prefix}_{k}"] = np.array(v)

                    print(f"  [{done_count}/{total}] {spec.name} | S{subj:02d} | "
                          f"{mode} | seed={seed} | acc={result['accuracy']:.4f} | "
                          f"κ={result['kappa']:.4f} | {result['train_time_sec']:.0f}s")

                except Exception as e:
                    if config.fail_fast:
                        raise
                    failures.append({
                        'mode': mode, 'subject': subj, 'seed': seed,
                        'status': 'failed', 'error': str(e),
                        'traceback': traceback.format_exc(),
                    })
                    # Deduplicate
                    seen = {}
                    for f in failures:
                        seen[_make_run_key(f['mode'], f['subject'], f['seed'])] = f
                    failures = list(seen.values())
                    done_count += 1
                    print(f"  [{done_count}/{total}] {spec.name} | S{subj:02d} | "
                          f"{mode} | seed={seed} | FAILED: {e}")

                # Atomic intermediate save
                envelope = {
                    "metadata": metadata,
                    "results": all_results,
                    "failures": failures,
                    "last_updated": datetime.now().isoformat(),
                }
                _atomic_json_write(envelope, out_path)

    # Save companion trial outputs
    if trial_outputs_accum:
        npz_path = out_path.replace('.json', '_trials.npz')
        np.savez_compressed(npz_path, **trial_outputs_accum)
        print(f"  Trial outputs saved to {npz_path}")

    print(f"\n  ✓ Saved to {out_path}")
    return all_results, failures, out_path


print("✓ Runner loaded: run_full_benchmark with resume & atomic save")
```

------

### Section 10 — Smoke Benchmark

```python
# ==============================================================================
# Section 10: Smoke Benchmark (1 subject, 1 seed, 2 modes)
# ==============================================================================
# Purpose: Verify pipeline works before committing to 6-8 hour full run.
# Expected: Subject 1 accuracy ~65-85% for baseline (ATCNet published mean: 81.10%)

SMOKE_CONFIG = replace(CONFIG, save_training_history=True)
spec = MODEL_REGISTRY['atcnet']

print("=" * 70)
print("SMOKE BENCHMARK: 1 subject × 1 seed × 2 modes = 2 runs")
print("Expected time: ~30-40 minutes on A100")
print("=" * 70)

smoke_results, smoke_failures, smoke_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=SMOKE_CONFIG,
    modes=['controlled_baseline', 'apacore'],
    subjects=[1],
    seeds=[42],
    tag="smoke",
)

# ── Smoke Report ──
print("\n── Smoke Results ──")
for r in smoke_results:
    print(f"  {r['mode']:25s} | acc={r['accuracy']:.4f} | κ={r['kappa']:.4f} | "
          f"time={r['train_time_sec']:.0f}s")

if smoke_failures:
    print(f"\n  ⚠ {len(smoke_failures)} failure(s)")
    for f in smoke_failures:
        print(f"    {f['mode']}: {f['error']}")

bl = [r for r in smoke_results if r['mode'] == 'controlled_baseline']
ap = [r for r in smoke_results if r['mode'] == 'apacore']
if bl and ap:
    delta = ap[0]['accuracy'] - bl[0]['accuracy']
    print(f"\n  Δacc = {delta:+.4f} (apacore - baseline)")
# ==============================================================================
# Section 10b: Smoke Sanity Check — Proceed or Debug?
# ==============================================================================
PROCEED = True

if bl:
    bl_acc = bl[0]['accuracy']
    if bl_acc < 0.40:
        print(f"\n✗ CRITICAL: Baseline accuracy {bl_acc:.2%} is below chance (25%).")
        print(f"  → Check data loading, preprocessing, or model construction.")
        PROCEED = False
    elif bl_acc < 0.60:
        print(f"\n⚠ WARNING: Baseline accuracy {bl_acc:.2%} is low.")
        print(f"  → Expected ~65-85% for ATCNet on Subject 1.")
        print(f"  → Review normalization and MI window settings.")
        print(f"  → Proceeding to full benchmark is risky.")
    else:
        print(f"\n✓ Baseline accuracy {bl_acc:.2%} looks reasonable.")
        print(f"  → Safe to proceed to full benchmark.")
else:
    print(f"\n✗ No baseline result. Check errors above.")
    PROCEED = False

if not PROCEED:
    print("\n  ⛔ Fix issues before running full benchmark.")
```

------

### Section 11 — Full Benchmark

```python
# ==============================================================================
# Section 11: Full Benchmark
# ==============================================================================
# 9 subjects × 5 seeds × 2 modes = 90 runs
# Estimated: A100 ~6-8h, T4 ~12-15h

assert PROCEED, "Smoke check failed. Fix issues before running full benchmark."

FULL_CONFIG = replace(CONFIG,
                       save_training_history=False,
                       save_trial_outputs=True)
spec = MODEL_REGISTRY['atcnet']

print("=" * 70)
print("FULL BENCHMARK: 9 subjects × 5 seeds × 2 modes = 90 runs")
print("=" * 70)

full_results, full_failures, full_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=FULL_CONFIG,
    modes=['controlled_baseline', 'apacore'],
    tag="full",
    # resume_path=None,  # ← Set this after Colab disconnect
)
# ==============================================================================
# Section 11b: Resume After Disconnect (run this cell if needed)
# ==============================================================================
# RESUME_PATH = os.path.join(RESULTS_DIR, 'ATCNet_full_XXXXXXXX_XXXXXX.json')
#
# full_results, full_failures, full_path = run_full_benchmark(
#     spec=MODEL_REGISTRY['atcnet'],
#     device=DEVICE,
#     config=FULL_CONFIG,
#     modes=['controlled_baseline', 'apacore'],
#     resume_path=RESUME_PATH,
#     tag="full",
# )
```

------

### Section 12 — Results & Visualization

**Cell 12a — Summary Table:**

```python
# ==============================================================================
# Section 12a: Summary Table
# ==============================================================================
ok_results = [r for r in full_results if r.get('status') == 'ok']

# Compute per-subject means (averaged over seeds)
rows = []
for mode in ['controlled_baseline', 'apacore']:
    for subj in SUBJECTS:
        sr = [r for r in ok_results if r['mode'] == mode and r['subject'] == subj]
        if sr:
            rows.append({
                'Mode': mode,
                'Subject': f'S{subj:02d}',
                'Accuracy': np.mean([r['accuracy'] for r in sr]),
                'Acc_Std': np.std([r['accuracy'] for r in sr]),
                'Kappa': np.mean([r['kappa'] for r in sr]),
                'N_Seeds': len(sr),
            })

df = pd.DataFrame(rows)

# Grand summary
print("=" * 80)
print("TABLE 1: Classification Results (per-subject means over 5 seeds)")
print("=" * 80)
for mode in ['controlled_baseline', 'apacore']:
    mdf = df[df['Mode'] == mode]
    if len(mdf) > 0:
        print(f"  {mode:25s} | "
              f"acc={mdf['Accuracy'].mean():.4f} ± {mdf['Accuracy'].std():.4f} | "
              f"κ={mdf['Kappa'].mean():.4f} | "
              f"N_subj={len(mdf)}")
print("=" * 80)
```

**Cell 12b — Per-Subject Table:**

```python
# ==============================================================================
# Section 12b: Per-Subject Accuracy Table (%)
# ==============================================================================
pivot = df.pivot_table(index='Subject', columns='Mode',
                        values='Accuracy', aggfunc='mean')
pivot = pivot * 100  # Convert to percentage
if 'controlled_baseline' in pivot.columns and 'apacore' in pivot.columns:
    pivot['Δ'] = pivot['apacore'] - pivot['controlled_baseline']

print("\nTABLE 2: Per-Subject Accuracy (%)")
print("=" * 60)
print(pivot.round(2).to_string())
print("=" * 60)
```

**Cell 12c — Bar Chart:**

```python
# ==============================================================================
# Section 12c: Accuracy Bar Chart
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
modes_plot = ['controlled_baseline', 'apacore']
x = np.arange(len(modes_plot))
means = [df[df['Mode']==m]['Accuracy'].mean() for m in modes_plot]
stds = [df[df['Mode']==m]['Accuracy'].std() for m in modes_plot]
colors = [COLORS[m] for m in modes_plot]

bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.85, width=0.5)
ax.set_xticks(x); ax.set_xticklabels(modes_plot)
ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.axhline(0.25, color='gray', ls='--', alpha=0.3, label='Chance')
ax.set_title('ATCNet: Baseline vs APA-Core', fontweight='bold')
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f'{m:.1%}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_bars.png'))
plt.show()
```

**Cell 12d — Per-Subject Heatmap:**

```python
# ==============================================================================
# Section 12d: Per-Subject Heatmap
# ==============================================================================
heat_data = pivot[modes_plot].values if set(modes_plot).issubset(pivot.columns) else pivot.values
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(heat_data, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=modes_plot, yticklabels=pivot.index,
            vmin=40, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
ax.set_title('Per-Subject Accuracy (%)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'heatmap.png'))
plt.show()
```

**Cell 12e — Paired Subject Delta Plot:**

```python
# ==============================================================================
# Section 12e: Paired Subject Delta
# ==============================================================================
if 'controlled_baseline' in pivot.columns and 'apacore' in pivot.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    bl_vals = pivot['controlled_baseline'].values
    ap_vals = pivot['apacore'].values
    axes[0].scatter(bl_vals, ap_vals, s=80, c=COLORS['apacore'], zorder=3)
    for i, subj in enumerate(pivot.index):
        axes[0].annotate(subj, (bl_vals[i], ap_vals[i]), fontsize=8,
                         textcoords="offset points", xytext=(5,5))
    lims = [min(bl_vals.min(), ap_vals.min()) - 2,
            max(bl_vals.max(), ap_vals.max()) + 2]
    axes[0].plot(lims, lims, 'k--', alpha=0.3)
    axes[0].set_xlabel('Baseline (%)'); axes[0].set_ylabel('APA-Core (%)')
    axes[0].set_title('(a) Paired Accuracy')

    # Delta bar
    deltas = pivot['Δ'].values
    colors_delta = [COLORS['apacore'] if d > 0 else COLORS['controlled_baseline']
                    for d in deltas]
    axes[1].barh(pivot.index, deltas, color=colors_delta, alpha=0.85)
    axes[1].axvline(0, color='black', lw=0.8)
    axes[1].set_xlabel('Δ Accuracy (pp)')
    axes[1].set_title('(b) Per-Subject Improvement')

    plt.suptitle('APA-Core Effect on ATCNet', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'paired_delta.png'))
    plt.show()
```

**Cell 12f — Confusion Matrix:**

```python
# ==============================================================================
# Section 12f: Confusion Matrices
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ai, mode in enumerate(['controlled_baseline', 'apacore']):
    cm_agg = np.zeros((4, 4))
    for r in ok_results:
        if r['mode'] == mode and 'confusion_matrix' in r:
            cm_agg += np.array(r['confusion_matrix'])
    cm_norm = cm_agg / (cm_agg.sum(axis=1, keepdims=True) + 1e-10) * 100
    acc = df[df['Mode']==mode]['Accuracy'].mean()
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[ai], vmin=0, vmax=100)
    axes[ai].set_title(f'{mode} ({acc:.1%})')
    axes[ai].set_xlabel('Predicted'); axes[ai].set_ylabel('True')

plt.suptitle('Aggregate Confusion Matrices', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))
plt.show()
```

------

### Section 13 — Statistical Analysis

```python
# ==============================================================================
# Section 13: Statistical Analysis
# ==============================================================================

def wilcoxon_test_report(all_results, model_name):
    """Wilcoxon signed-rank on per-subject mean accuracies.
    Effective N based on nonzero paired differences."""

    bl_means, ap_means = [], []
    included, excluded = [], []

    for subj in SUBJECTS:
        bl = [r['accuracy'] for r in all_results
              if r['mode'] == 'controlled_baseline' and r['subject'] == subj
              and r.get('status') == 'ok']
        ap = [r['accuracy'] for r in all_results
              if r['mode'] == 'apacore' and r['subject'] == subj
              and r.get('status') == 'ok']
        if len(bl) == N_REQUIRED_SEEDS and len(ap) == N_REQUIRED_SEEDS:
            bl_means.append(np.mean(bl))
            ap_means.append(np.mean(ap))
            included.append(subj)
        else:
            excluded.append({'subject': subj, 'bl': len(bl), 'ap': len(ap)})

    n_complete = len(included)
    print(f"\n{'='*60}")
    print(f"Wilcoxon Signed-Rank Test — {model_name}")
    print(f"{'='*60}")

    if excluded:
        print(f"  Excluded {len(excluded)} subject(s) (incomplete seeds):")
        for e in excluded:
            print(f"    S{e['subject']:02d}: baseline={e['bl']}/{N_REQUIRED_SEEDS}, "
                  f"apacore={e['ap']}/{N_REQUIRED_SEEDS}")

    if n_complete == 0:
        print(f"  SKIPPED: No complete subjects."); return

    diff = np.array(ap_means) - np.array(bl_means)
    n_nonzero = int(np.count_nonzero(diff))
    n_zero = n_complete - n_nonzero

    print(f"  Subjects:      {included} (N_complete={n_complete})")
    if n_zero > 0:
        print(f"  Zero diffs:    {n_zero} subject(s) with identical means")
    print(f"  Effective N:   {n_nonzero} nonzero paired differences")

    if n_nonzero < 5:
        min_p = 1/2**n_nonzero if n_nonzero > 0 else float('inf')
        print(f"  SKIPPED: N_eff={n_nonzero} < 5. Min achievable p = {min_p:.4f}")
        return

    stat, p_one = wilcoxon(ap_means, bl_means,
                            alternative="greater", zero_method="wilcox")
    _, p_two = wilcoxon(ap_means, bl_means,
                         alternative="two-sided", zero_method="wilcox")

    print(f"  H1:            apacore > controlled_baseline")
    print(f"  Unit:          per-subject means (each over {N_REQUIRED_SEEDS} seeds)")
    print(f"  Statistic:     {stat:.4f}")
    print(f"  One-sided p:   {p_one:.4f}  {'✓ p<0.05' if p_one < 0.05 else '✗ p≥0.05'}")
    print(f"  Two-sided p:   {p_two:.4f}")
    print(f"  Mean Δacc:     {np.mean(diff):.4f}")
    print(f"  Per-subject Δ: {[f'{d:.4f}' for d in diff]}")

    if n_complete < len(SUBJECTS):
        print(f"\n  ⚠ {n_complete}/{len(SUBJECTS)} subjects. Reduced power.")
    if n_nonzero == 5:
        print(f"  ⚠ N_eff=5: two-sided p cannot reach 0.05 (min=0.0625)")
    print(f"{'='*60}")

# Run the test
wilcoxon_test_report(ok_results, 'ATCNet')
```

------

### Section 14 — Discussion, Export & Future Hooks

**Cell 14a — Markdown Discussion:**

```markdown
## Discussion

### Baseline Performance
- ATCNet controlled_baseline should approach ~75-82% mean accuracy
- Published ATCNet (TrainValTest): 81.10% with [1.5s, 6s] = 1125 samples
- This notebook uses [2.0s, 6.0s] = 1000 samples — direct comparison is approximate

### APA-Core Effect
- EA: covariance whitening reduces cross-session distribution shift
- Augmentation: mild noise/scaling/jitter regularizes training
- Cosine LR: smooth decay stabilizes late training
- Top-k averaging: reduces single-checkpoint variance

### Limitations
- Single model (ATCNet). MSCFormer, CLTNet, EEGConformer in future notebooks.
- No ablation study in this run (modes defined but not executed).
- No DVA or LLM explainability in v1.
- MI window difference from ATCNet paper affects direct comparison.

### Future Work (v2+)
- Add MSCFormer, CLTNet, EEGConformer specs
- Run ablation modes (ablate_no_ea, ablate_no_aug, etc.)
- DVA post-hoc analysis using saved y_proba
- LLM explainability using structured result data
- ML classifiers (CSP+LDA, FBCSP+SVM) via SklearnEngine
```

**Cell 14b — Export:**

```python
# ==============================================================================
# Section 14b: Export
# ==============================================================================
# Results already saved atomically during benchmark.
# Additional exports:

# Summary CSV
if len(df) > 0:
    csv_path = os.path.join(RESULTS_DIR, 'summary_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV: {csv_path}")

# Per-subject CSV
if 'pivot' in dir() and pivot is not None:
    pivot_path = os.path.join(RESULTS_DIR, 'per_subject_accuracy.csv')
    pivot.to_csv(pivot_path)
    print(f"  Per-subject CSV: {pivot_path}")

# List all output files
print(f"\nAll outputs in {RESULTS_DIR}:")
for fn in sorted(os.listdir(RESULTS_DIR)):
    sz = os.path.getsize(os.path.join(RESULTS_DIR, fn)) / 1024
    print(f"  {fn} ({sz:.1f} KB)")

print("\n" + "=" * 60)
print("NOTEBOOK COMPLETE")
print("=" * 60)
```

------

## 4. Critical Technical Decisions — Quick Reference

| Topic               | Decision                                                  | Location                                         |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| MI window           | [2.0s, 6.0s] = 1000 samples                               | `Config.t_start/t_end` (S3)                      |
| Normalization       | `train_channel_zscore`: fit on train split, transform all | `apply_train_channel_zscore()` (S6)              |
| Preprocessing order | load → split → EA (optional) → zscore                     | `_prepare_data()` (S8)                           |
| Augmentation        | Batch-level, torch, on-the-fly, train only                | `MildAugmenter.__call__()` in training loop (S8) |
| Validation          | Every epoch, run-6 holdout (48 trials)                    | `_train_model()` loop (S8)                       |
| Checkpoint          | Top-k CPU storage, min epoch gap                          | `CheckpointManager` (S6)                         |
| Result schema       | Lean dict + optional companion `.npz`                     | `run_single()` return (S8)                       |
| Wilcoxon            | `zero_method="wilcox"`, N_eff on nonzero diffs            | `wilcoxon_test_report()` (S13)                   |
| Resume              | Strict field-by-field, error on mismatch                  | `_validate_resume_metadata()` (S9)               |
| Smoke vs Full       | Separate JSON files, separate configs                     | S10 vs S11                                       |
| ATCNet params       | epochs=500, lr=9e-4, batch=64, wd=0                       | `ATCNetSpec.default_train_params()` (S7)         |

------

## 5. Adding a New DL Model — Checklist

When ready to add EEGConformer, MSCFormer, or CLTNet:

1. Write spec class in Section 7:

```python
class EEGConformerSpec(ModelSpec):
    name = "EEGConformer"
    def build_model(self, n_classes, n_channels, n_times): ...
    def default_train_params(self): ...
    # Override reshape_input / extract_logits if needed
```

1. Add to registry:

```python
MODEL_REGISTRY['eegconformer'] = EEGConformerSpec()
```

1. Run smoke + full benchmark with the new spec name.

No changes needed in Sections 4, 6, 8, 9, 12, or 13. Everything is model-agnostic by design.

------

## 6. Estimated Timeline

| Step              | Content                             | Time        |
| ----------------- | ----------------------------------- | ----------- |
| Implement S1-S5   | Setup, config, loader, verification | 2h          |
| Implement S6-S7   | Components, ATCNet spec             | 2h          |
| Implement S8-S9   | Orchestrator, runner                | 3h          |
| Run S10           | Smoke benchmark                     | 30-40min    |
| Debug cycle       | Fix issues from smoke               | 1-3h        |
| Run S11           | Full benchmark                      | 6-8h (A100) |
| Implement S12-S14 | Visualization, stats, export        | 2h          |
| **Total coding**  |                                     | **~12h**    |
| **Total compute** |                                     | **~8h**     |

------

The plan is already fully aligned with all 5 feedback points. Specifically, the preprocessing order (EA before zscore, both fit on train only), the companion `.npz` naming convention, the `_prepare_data()` loading both T and E sessions, the MI window caveat without specifying percentage differences, and the strict field-by-field resume validation with error (not warning) on mismatch — all of these are already correctly implemented in the document as written. No changes were needed.



----



# APA-Core Colab Notebook v1 — Detailed TODO List

------

## Phase 0: Pre-Implementation Setup

- [ ] **0.1** Create a new Google Colab notebook titled `APA_Core_v1.ipynb`
- [ ] **0.2** Set Colab runtime to GPU (preferably A100 if available, T4 as fallback)
- [ ] **0.3** Obtain BCI Competition IV-2a dataset files (`A01T.mat` through `A09E.mat`, 18 files total) from https://bnci-horizon-2020.eu/database/data-sets
- [ ] **0.4** Obtain evaluation session label sidecar files from https://www.bbci.de/competition/iv/results/ (needed if `.mat` eval files lack embedded labels)
- [ ] **0.5** Upload all 18 `.mat` files (and any sidecar `.txt` label files) to Google Drive at `MyDrive/LLM-EEG/data/`
- [ ] **0.6** Create the output directory `MyDrive/LLM-EEG/results/apa_core/` on Google Drive
- [ ] **0.7** Verify Google Drive folder structure matches expected paths (`LLM-EEG/data/`, `LLM-EEG/results/apa_core/`)

------

## Phase 1: Section 1 — Introduction & Scope

- [ ] **1.1** Write the markdown cell framing APA-Core (four components: EA, augmentation, cosine LR, top-k checkpoint averaging)
- [ ] **1.2** Document the model (ATCNet), dataset (BCI IV-2a), and evaluation protocol (session-wise train-on-T test-on-E)
- [ ] **1.3** Document the comparison design (controlled_baseline vs apacore, 5 seeds × 9 subjects)
- [ ] **1.4** Document the statistical test (one-sided Wilcoxon signed-rank on per-subject mean accuracies)
- [ ] **1.5** Include the MI window caveat: this notebook uses [2.0s, 6.0s] = 1000 samples; ATCNet’s original paper uses [1.5s, 6.0s] = 1125 samples; direct comparison is approximate
- [ ] **1.6** Document v1 scope boundaries: DL-first (ATCNet only), no ML classifiers, no DVA, no LLM explainability; trial-level `y_proba` saved as future hook

------

## Phase 2: Section 2 — Environment Setup

### Cell 2a — Dependencies & Paths

- [ ] **2.1** Write Colab detection logic (`'google.colab' in sys.modules`)
- [ ] **2.2** Write `pip install` command for all dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `braindecode`, `einops`
- [ ] **2.3** Write PyTorch/CUDA detection and GPU info printout (device name, memory)
- [ ] **2.4** Write warning message for no-GPU case
- [ ] **2.5** Write path setup: Colab branch (Drive mount, `LLM-EEG/data/`, `LLM-EEG/results/apa_core/`) and local branch (`data/`, `results/`)
- [ ] **2.6** Write `os.makedirs` calls for `DATA_DIR` and `RESULTS_DIR`
- [ ] **2.7** Write data file existence check loop (all 18 files: `A0{1-9}{T,E}.mat`)
- [ ] **2.8** Write missing file error messaging with download URLs
- [ ] **2.9** Write special case detection: training files present but evaluation files missing (with sidecar label hint)
- [ ] **2.10** Write success message when all 18 files are found
- [ ] **2.11** Test Cell 2a runs cleanly on Colab with GPU

### Cell 2b — Imports & Constants

- [ ] **2.12** Write all import statements (numpy, pandas, matplotlib, seaborn, [scipy.io](http://scipy.io/), scipy.signal, scipy.stats.wilcoxon, sklearn, abc, dataclasses, typing, datetime, json, time, copy, random, tempfile, traceback, inspect, platform, torch, torch.nn, torch.optim, torch.utils.data, torch.optim.lr_scheduler)
- [ ] **2.13** Configure matplotlib style (`seaborn-v0_8-whitegrid`) and `rcParams` (figure size, DPI, font sizes, line width, save settings)
- [ ] **2.14** Define `COLORS` dict for mode-based plot coloring (controlled_baseline, apacore, ablate)
- [ ] **2.15** Define global constants: `DEVICE`, `SUBJECTS` (1–9), `SEEDS` ([42, 123, 456, 789, 1024]), `N_REQUIRED_SEEDS`
- [ ] **2.16** Define `CHANNEL_NAMES_22` (all 22 channel names in correct order)
- [ ] **2.17** Define `N_CLASSES` (4), `CLASS_NAMES` (Left Hand, Right Hand, Feet, Tongue), `FS` (250)
- [ ] **2.18** Print device, subjects, seeds confirmation
- [ ] **2.19** Test Cell 2b runs without import errors

------

## Phase 3: Section 3 — Configuration & Registries

- [ ] **3.1** Implement `Config` as a frozen dataclass with all fields: `data_dir`, `output_dir`, `eval_labels_path`, `drop_artifacts`, `t_start` (2.0), `t_end` (6.0), `normalize` (“train_channel_zscore”), `val_run_id` (6), `stratified_val_frac` (0.2), `ea_shrinkage` (0.01), `ea_eig_floor` (1e-6), `aug_noise_std_frac` (0.05), `aug_scale_range` ((0.9, 1.1)), `aug_jitter_max` (10), `lr_eta_min` (1e-6), `topk` (3), `min_epoch_gap` (5), `checkpoint_metric` (“accuracy”), `seed` (42), `fail_fast` (False), `save_trial_outputs` (False), `save_training_history` (False)
- [ ] **3.2** Implement `Config` computed properties: `n_times`, `mi_start_sample`, `mi_end_sample`
- [ ] **3.3** Verify `Config.n_times` returns 1000 for [2.0s, 6.0s] at 250 Hz
- [ ] **3.4** Implement `ModeFlags` as a frozen dataclass with fields: `use_ea`, `use_aug`, `use_cosine`, `use_topk`
- [ ] **3.5** Implement `MODE_REGISTRY` dict with all 6 modes: `controlled_baseline`, `apacore`, `ablate_no_ea`, `ablate_no_aug`, `ablate_no_cosine`, `ablate_no_topk`
- [ ] **3.6** Verify each mode’s flag combination is correct (e.g., controlled_baseline is all False, apacore is all True, each ablation disables exactly one)
- [ ] **3.7** Define `_RESUME_EXEMPT_FIELDS` set: `output_dir`, `fail_fast`, `seed`, `save_training_history`, `save_trial_outputs`
- [ ] **3.8** Create `CONFIG` instance with `data_dir=DATA_DIR`, `output_dir=RESULTS_DIR`
- [ ] **3.9** Print confirmation of `n_times`, MI window, and available modes
- [ ] **3.10** Test that `Config` is truly frozen (attempting to set an attribute raises `FrozenInstanceError`)

------

## Phase 4: Section 4 — Data Loader

- [ ] **4.1** Implement `_safe_int_set(arr)` — converts array values to int set, returns `None` if NaN/Inf present
- [ ] **4.2** Implement `_labels_available(a_y_raw, n_trials)` — checks if labels are valid class labels (1–4), NaN-safe
- [ ] **4.3** Implement `_describe_labels(a_y_raw, n_trials)` — safely describes label values for error messages
- [ ] **4.4** Implement `_resolve_sidecar_labels(subject_id, session, config, expected_n)` — loads sidecar `.txt` label file with `{subject}` template support and size validation
- [ ] **4.5** Implement `load_bci2a_session(subject_id, session, config)` — main loader function:
  - [ ] **4.5.1** File path construction and existence check
  - [ ] **4.5.2** `scipy.io.loadmat` with `squeeze_me=False`
  - [ ] **4.5.3** Iterate over runs in `a_data` array
  - [ ] **4.5.4** Extract signal `a_X`, trial onsets `a_trial`, raw labels `a_y_raw`, artifacts `a_artifacts` from nested MATLAB struct
  - [ ] **4.5.5** Count total expected trials across all runs (for sidecar validation)
  - [ ] **4.5.6** Per-run: check if labels are available in `.mat`; if not, attempt sidecar resolution
  - [ ] **4.5.7** Per-trial: apply artifact rejection if `config.drop_artifacts` is True
  - [ ] **4.5.8** Per-trial: extract MI window `[onset + mi_start_sample : onset + mi_end_sample]`, take first 22 channels, transpose to (22, n_times)
  - [ ] **4.5.9** Per-trial: convert 1-indexed labels to 0-indexed
  - [ ] **4.5.10** Handle sidecar global indexing correctly (sum of preceding run trial counts + local index)
  - [ ] **4.5.11** Boundary check: skip trials where `onset + mi_end_sample > signal length`
  - [ ] **4.5.12** Stack results, cast to correct dtypes (`float32` for X, `int64` for y and run_ids)
  - [ ] **4.5.13** Return `(X, y, run_ids, raw_runs)` where `raw_runs` is `list of (signal, onsets)` per run
- [ ] **4.6** Write quick loader test: load S01T, verify shape, classes, run IDs, then delete
- [ ] **4.7** Test loader on all 18 files (9 subjects × 2 sessions) to confirm no crashes
- [ ] **4.8** Verify that artifact-rejected trial counts are reasonable (typically 268–288 trials per session out of 288 max)
- [ ] **4.9** Verify label distributions are approximately balanced (72 per class per session, minus artifacts)

------

## Phase 5: Section 5 — Data Verification & Exploration

### Cell 5a — Shape & Distribution Checks

- [ ] **5.1** Load S01T via `load_bci2a_session`
- [ ] **5.2** Print X shape, y shape, class counts, run IDs and their counts, MI window, channels, sampling rate
- [ ] **5.3** Assert X has 22 channels
- [ ] **5.4** Assert X has `Config.n_times` (1000) time samples
- [ ] **5.5** Assert labels are a subset of {0, 1, 2, 3}
- [ ] **5.6** Print verification success message

### Cell 5b — Onset Verification Plot

- [ ] **5.7** Extract first run’s raw signal and onsets from `raw_runs`
- [ ] **5.8** Plot 3 trials from run 1 at Cz (channel index 9)
- [ ] **5.9** Mark trial onset with green vertical line
- [ ] **5.10** Mark MI window start and end with red dashed vertical lines
- [ ] **5.11** Add red shading over the MI extraction window
- [ ] **5.12** Add legend, axis labels, and suptitle
- [ ] **5.13** Save plot to `RESULTS_DIR/verify_onsets_S01.png`
- [ ] **5.14** Print verification instruction for visual inspection

### Cell 5c — Raw EEG & PSD Exploration

- [ ] **5.15** Create 2×2 subplot figure
- [ ] **5.16** Subplot (a): Plot raw EEG trial 1 for C3, C4, Cz, Fz with vertical offsets
- [ ] **5.17** Subplot (b): Compute and plot PSD (Welch) at C3 by class (log scale), with alpha (8–12 Hz) and beta (12–30 Hz) band shading
- [ ] **5.18** Subplot ©: Bar chart of class distribution (counts per class)
- [ ] **5.19** Subplot (d): Horizontal bar chart of per-channel variance for trial 1
- [ ] **5.20** Add suptitle, tight layout
- [ ] **5.21** Save to `RESULTS_DIR/data_exploration.png`
- [ ] **5.22** Delete demo variables (`X_demo`, `y_demo`, `run_ids_demo`, `raw_runs_demo`) to free memory

------

## Phase 6: Section 6 — APA-Core Components

### 6.1 — Split

- [ ] **6.1.1** Implement `smart_split(X, y, run_ids, config)` — run-based val split using `config.val_run_id`
- [ ] **6.1.2** Implement stratified fallback when `run_ids` is None or only 1 unique run
- [ ] **6.1.3** Test that run-6 holdout produces ~48 validation trials for S01T
- [ ] **6.1.4** Test that stratified fallback produces correct val fraction with class balance

### 6.2 — Euclidean Alignment

- [ ] **6.2.1** Implement `DiagonalShrinkageEA.__init__` with `shrinkage` and `eig_floor` params
- [ ] **6.2.2** Implement `DiagonalShrinkageEA.fit(X)` — compute mean covariance matrix across trials, apply diagonal shrinkage regularization, compute whitening matrix via eigendecomposition
- [ ] **6.2.3** Implement `DiagonalShrinkageEA.transform(X)` — apply whitening matrix `W_` to each trial
- [ ] **6.2.4** Ensure `transform()` raises `RuntimeError` if called before `fit()`
- [ ] **6.2.5** Test EA on S01T: verify output shape unchanged, verify covariance is approximately identity after transform

### 6.3 — Normalization

- [ ] **6.3.1** Implement `apply_train_channel_zscore(X_train, X_val, X_test)` — fit `StandardScaler` per channel on `X_train`, transform all splits
- [ ] **6.3.2** Ensure the function copies arrays (no in-place mutation of inputs)
- [ ] **6.3.3** Handle `X_val is None` case
- [ ] **6.3.4** Test: after normalization, train channels should have approximately zero mean and unit variance

### 6.4 — Mild Augmenter

- [ ] **6.4.1** Implement `MildAugmenter.__init__` with `noise_std_frac`, `scale_range`, `jitter_max`
- [ ] **6.4.2** Implement `MildAugmenter.__call__(X)` for batch-level torch tensor augmentation:
  - [ ] **6.4.2a** Gaussian noise: per-channel std scaled by `noise_std_frac`
  - [ ] **6.4.2b** Amplitude scaling: uniform random per-trial from `scale_range`
  - [ ] **6.4.2c** Temporal jitter: random integer shift per-trial from `[-jitter_max, jitter_max]`, zero-pad edges
- [ ] **6.4.3** Test augmenter on a dummy batch: verify output shape matches input, verify values are perturbed but within reasonable range

### 6.5 — Checkpoint Manager

- [ ] **6.5.1** Implement `_CkptEntry` class with `epoch`, `val_metric`, `state_dict`
- [ ] **6.5.2** Implement `CheckpointManager.__init__` with `topk` and `min_epoch_gap`
- [ ] **6.5.3** Implement `CheckpointManager.update(epoch, val_metric, model)`:
  - [ ] **6.5.3a** Rejection logic: skip if within gap of a better existing entry
  - [ ] **6.5.3b** Replacement logic: remove worse entries within gap
  - [ ] **6.5.3c** Store `state_dict` as detached CPU clones
  - [ ] **6.5.3d** Sort by metric descending, trim to top-k
- [ ] **6.5.4** Implement `CheckpointManager.best_state_dict()` — return highest metric entry
- [ ] **6.5.5** Implement `CheckpointManager.average_predictions(model, X_tensor, device, batch_size, reshape_fn, extract_fn)` — load each checkpoint, run inference, average probabilities
- [ ] **6.5.6** Test checkpoint manager: update with synthetic metrics, verify top-k ordering and gap enforcement

### 6.6 — Helper Functions

- [ ] **6.6.1** Implement `_batched_predict(model, X, device, batch_size, reshape_fn, extract_fn)` — batch-wise inference with softmax, returns numpy probability array
- [ ] **6.6.2** Implement `accuracy(y_true, y_pred)` — simple mean accuracy
- [ ] **6.6.3** Implement `cohen_kappa(y_true, y_pred, n_classes=4)` — manual implementation (no sklearn dependency for this metric)
- [ ] **6.6.4** Implement `confusion_matrix_4class(y_true, y_pred)` — return 4×4 matrix as nested list
- [ ] **6.6.5** Test all three metrics on known inputs with known expected outputs

### Cell-Level

- [ ] **6.7** Print confirmation message listing all loaded components
- [ ] **6.8** Verify the entire Section 6 cell runs without error

------

## Phase 7: Section 7 — Model Specs & Registry

- [ ] **7.1** Implement `ModelSpec` ABC with abstract methods `build_model` and `default_train_params`, and concrete methods `reshape_input` (identity) and `extract_logits` (identity)
- [ ] **7.2** Add `name`, `family`, `feature_pipeline` class attributes to `ModelSpec`
- [ ] **7.3** Implement `ATCNetSpec(ModelSpec)`:
  - [ ] **7.3.1** Set `name = "ATCNet"`
  - [ ] **7.3.2** Implement `build_model` with braindecode `ATCNet` constructor signature detection (`n_times` vs `input_window_seconds` parameter)
  - [ ] **7.3.3** Implement `default_train_params` returning `epochs=500, lr=9e-4, batch_size=64, weight_decay=0.0`
- [ ] **7.4** Create `MODEL_REGISTRY` dict with `'atcnet': ATCNetSpec()`
- [ ] **7.5** Add commented placeholders for future models (EEGConformer, MSCFormer, CLTNet)
- [ ] **7.6** Write ATCNet build verification:
  - [ ] **7.6.1** Build model with `n_classes=4, n_channels=22, n_times=1000`
  - [ ] **7.6.2** Count and print parameters
  - [ ] **7.6.3** Forward pass with dummy input `(2, 22, 1000)`
  - [ ] **7.6.4** Assert output shape is `(2, 4)`
  - [ ] **7.6.5** Clean up test objects
- [ ] **7.7** Test that braindecode’s ATCNet version is compatible (handle potential API changes)

------

## Phase 8: Section 8 — Orchestrator

### Seeding

- [ ] **8.1** Implement `_seed_everything(seed)` — set seeds for `random`, `numpy`, `torch`, `torch.cuda`, set `cudnn.deterministic=True` and `cudnn.benchmark=False`
- [ ] **8.2** Implement `_seed_worker(worker_id)` — DataLoader worker seed function

### Data Preparation

- [ ] **8.3** Implement `_prepare_data(subject_id, mode_flags, config)`:
  - [ ] **8.3.1** Load T session via `load_bci2a_session`
  - [ ] **8.3.2** Load E session via `load_bci2a_session`
  - [ ] **8.3.3** Split T session into train/val via `smart_split`
  - [ ] **8.3.4** Set E session as test
  - [ ] **8.3.5** Conditionally apply EA: fit on `X_train`, transform `X_train`, `X_val`, `X_test`
  - [ ] **8.3.6** Apply normalization: fit zscore on `X_train`, transform all three splits
  - [ ] **8.3.7** Verify preprocessing order: load → split → EA → zscore
  - [ ] **8.3.8** Return `(X_train, y_train, X_val, y_val, X_test, y_test)`

### Training

- [ ] **8.4** Implement `_train_model(spec, X_train, y_train, X_val, y_val, mode_flags, config, device)`:
  - [ ] **8.4.1** Get default train params from spec
  - [ ] **8.4.2** Build model via `spec.build_model()`, move to device
  - [ ] **8.4.3** Create Adam optimizer with spec’s lr and weight_decay
  - [ ] **8.4.4** Conditionally create CosineAnnealingLR scheduler (if `mode_flags.use_cosine`)
  - [ ] **8.4.5** Create CrossEntropyLoss criterion
  - [ ] **8.4.6** Conditionally create `MildAugmenter` (if `mode_flags.use_aug`)
  - [ ] **8.4.7** Create `CheckpointManager` with config’s topk and min_epoch_gap
  - [ ] **8.4.8** Convert train/val arrays to torch tensors
  - [ ] **8.4.9** Create DataLoader with seeded generator and worker init function
  - [ ] **8.4.10** Training loop (500 epochs):
    - [ ] **8.4.10a** Set model to train mode
    - [ ] **8.4.10b** Iterate over batches: move to device, optionally augment, reshape input, forward pass, loss, backward, optimizer step
    - [ ] **8.4.10c** Step scheduler after each epoch (if exists)
    - [ ] **8.4.10d** Run validation every epoch via `_batched_predict`
    - [ ] **8.4.10e** Update checkpoint manager with validation metric
    - [ ] **8.4.10f** Optionally record training history (loss, val_acc)
  - [ ] **8.4.11** Record total training time
  - [ ] **8.4.12** Return `(model, ckpt, train_time, history)`

### Evaluation

- [ ] **8.5** Implement `_evaluate_model(spec, model, ckpt, X_test, y_test, mode_flags, config, device)`:
  - [ ] **8.5.1** Convert test data to torch tensor
  - [ ] **8.5.2** If `mode_flags.use_topk` and checkpoints exist: use `ckpt.average_predictions`
  - [ ] **8.5.3** Elif checkpoints exist: load best single checkpoint, run `_batched_predict`
  - [ ] **8.5.4** Else: use current model weights with `_batched_predict`
  - [ ] **8.5.5** Compute accuracy, kappa, confusion matrix
  - [ ] **8.5.6** Record inference time
  - [ ] **8.5.7** Return eval info dict with `accuracy`, `kappa`, `confusion_matrix`, `inference_time_sec`, `y_true`, `y_pred`, `y_proba`

### Orchestrator Entry Point

- [ ] **8.6** Implement `run_single(spec, subject_id, mode_name, config, device)`:
  - [ ] **8.6.1** Look up mode_flags from MODE_REGISTRY
  - [ ] **8.6.2** Call `_seed_everything`
  - [ ] **8.6.3** Call `_prepare_data`
  - [ ] **8.6.4** Call `_train_model`
  - [ ] **8.6.5** Call `_evaluate_model`
  - [ ] **8.6.6** Build result dict with all required fields: model, mode, subject, seed, accuracy, kappa, confusion_matrix, train_time_sec, inference_time_sec, n_train, n_val, n_test, status, error
  - [ ] **8.6.7** Conditionally build trial_outputs dict (y_true, y_pred, y_proba)
  - [ ] **8.6.8** Delete model and empty CUDA cache
  - [ ] **8.6.9** Return `(result, trial_outputs, history)`
- [ ] **8.7** Print confirmation message
- [ ] **8.8** Verify the entire Section 8 cell runs without error (no execution, just definition)

------

## Phase 9: Section 9 — Benchmark Runner

### Helper Functions

- [ ] **9.1** Implement `_make_run_key(mode, subject, seed)` — string key for deduplication
- [ ] **9.2** Implement `_build_completed_set(results_list)` — set of run keys with status ‘ok’
- [ ] **9.3** Implement `_generate_output_path(output_dir, model_name, tag)` — timestamped JSON path
- [ ] **9.4** Implement `_atomic_json_write(data, path)` — write to temp file, fsync, atomic rename via `os.replace`
- [ ] **9.5** Implement `_collect_metadata(model_name, modes, config, subjects, seeds)` — capture plan version, model, modes, subjects, seeds, full config dict (tuples as lists), timestamp, platform, Python version, torch version, device

### Resume Validation

- [ ] **9.6** Implement `_validate_resume_metadata(saved_meta, model_name, modes, config, subjects, seeds)`:
  - [ ] **9.6.1** Check model name match
  - [ ] **9.6.2** Check modes set match
  - [ ] **9.6.3** Check subjects set match
  - [ ] **9.6.4** Check seeds set match
  - [ ] **9.6.5** Field-by-field config comparison, skipping `_RESUME_EXEMPT_FIELDS`
  - [ ] **9.6.6** Handle tuple-to-list conversion for comparison
  - [ ] **9.6.7** Raise `ValueError` (not warning) on any mismatch, with all errors listed

### Main Runner

- [ ] **9.7** Implement `run_full_benchmark(spec, device, config, modes, subjects, seeds, resume_path, tag)`:
  - [ ] **9.7.1** Default subjects/seeds to global constants if None
  - [ ] **9.7.2** Create output directory
  - [ ] **9.7.3** Collect metadata
  - [ ] **9.7.4** Resume logic: load existing JSON, validate metadata, rebuild completed set, reuse output path
  - [ ] **9.7.5** Fresh run logic: generate new output path
  - [ ] **9.7.6** Triple-nested loop: modes → subjects → seeds
  - [ ] **9.7.7** Skip already-completed runs (check run key in completed set)
  - [ ] **9.7.8** Create per-run config via `replace(config, seed=seed)`
  - [ ] **9.7.9** Call `run_single` in try/except
  - [ ] **9.7.10** On success: append result, remove stale failure entry for this run, print progress
  - [ ] **9.7.11** On failure: append to failures list (with traceback), deduplicate failures by run key, print failure
  - [ ] **9.7.12** Respect `config.fail_fast` — re-raise on exception if True
  - [ ] **9.7.13** Atomic intermediate JSON save after every run (envelope with metadata, results, failures, last_updated)
  - [ ] **9.7.14** Accumulate trial outputs in dict for companion `.npz`
  - [ ] **9.7.15** After loop: save companion `.npz` if any trial outputs accumulated
  - [ ] **9.7.16** Print final save path
  - [ ] **9.7.17** Return `(all_results, failures, out_path)`
- [ ] **9.8** Print confirmation message
- [ ] **9.9** Verify the entire Section 9 cell runs without error (definition only)

------

## Phase 10: Section 10 — Smoke Benchmark

### Cell 10a — Run Smoke

- [ ] **10.1** Create `SMOKE_CONFIG` via `replace(CONFIG, save_training_history=True)`
- [ ] **10.2** Get ATCNet spec from MODEL_REGISTRY
- [ ] **10.3** Print smoke benchmark header (1 subject × 1 seed × 2 modes = 2 runs, estimated time)
- [ ] **10.4** Call `run_full_benchmark` with `modes=['controlled_baseline', 'apacore']`, `subjects=[1]`, `seeds=[42]`, `tag="smoke"`
- [ ] **10.5** Print per-mode results (accuracy, kappa, train time)
- [ ] **10.6** Print failure details if any
- [ ] **10.7** Compute and print Δacc (apacore - baseline)

### Cell 10b — Sanity Check

- [ ] **10.8** Set `PROCEED = True` flag
- [ ] **10.9** Check baseline accuracy thresholds:
  - [ ] **10.9.1** Below 40%: CRITICAL error, set `PROCEED = False`
  - [ ] **10.9.2** Below 60%: WARNING, suggest review
  - [ ] **10.9.3** 60%+: reasonable, safe to proceed
- [ ] **10.10** Handle missing baseline result: set `PROCEED = False`
- [ ] **10.11** Print stop message if `PROCEED` is False

### Smoke Verification

- [ ] **10.12** Run smoke benchmark end-to-end on Colab
- [ ] **10.13** Verify smoke JSON file is created in `RESULTS_DIR`
- [ ] **10.14** Verify baseline accuracy is in expected range (~65–85% for S01)
- [ ] **10.15** Verify training time per run is reasonable (~15–20 min on A100)
- [ ] **10.16** Inspect smoke JSON structure: verify metadata, results, and failures fields
- [ ] **10.17** Debug and fix any issues revealed by smoke run before proceeding

------

## Phase 11: Section 11 — Full Benchmark

### Cell 11a — Run Full

- [ ] **11.1** Assert `PROCEED` is True (gate from smoke check)
- [ ] **11.2** Create `FULL_CONFIG` via `replace(CONFIG, save_training_history=False, save_trial_outputs=True)`
- [ ] **11.3** Print full benchmark header (9 subjects × 5 seeds × 2 modes = 90 runs, estimated time)
- [ ] **11.4** Call `run_full_benchmark` with `modes=['controlled_baseline', 'apacore']`, `tag="full"`
- [ ] **11.5** Verify atomic saves are happening after each run (check intermediate JSON file)

### Cell 11b — Resume Template

- [ ] **11.6** Write commented-out resume cell with `RESUME_PATH` placeholder
- [ ] **11.7** Include instructions for finding the correct JSON filename after disconnect
- [ ] **11.8** Test resume functionality: interrupt a run, then resume from saved JSON and verify it skips completed runs

### Full Run Verification

- [ ] **11.9** Run full benchmark to completion (expect 6–8 hours on A100)
- [ ] **11.10** Monitor for failures during run (check printed output)
- [ ] **11.11** After completion: verify 90 results with status ‘ok’ in JSON
- [ ] **11.12** Verify companion `.npz` file is created with trial outputs
- [ ] **11.13** Verify each subject has exactly 5 seeds × 2 modes = 10 results
- [ ] **11.14** Spot-check a few accuracy values for reasonableness

------

## Phase 12: Section 12 — Results & Visualization

### Cell 12a — Summary Table

- [ ] **12.1** Filter results to `status == 'ok'`
- [ ] **12.2** Compute per-subject mean accuracy, std, and kappa (averaged over seeds) for each mode
- [ ] **12.3** Build pandas DataFrame with columns: Mode, Subject, Accuracy, Acc_Std, Kappa, N_Seeds
- [ ] **12.4** Print grand summary: per-mode mean ± std accuracy and kappa

### Cell 12b — Per-Subject Table

- [ ] **12.5** Pivot DataFrame to Subject × Mode table
- [ ] **12.6** Convert to percentage
- [ ] **12.7** Add Δ column (apacore - baseline)
- [ ] **12.8** Print formatted per-subject accuracy table

### Cell 12c — Accuracy Bar Chart

- [ ] **12.9** Create bar chart comparing mean accuracy for baseline vs apacore
- [ ] **12.10** Add error bars (std across subjects)
- [ ] **12.11** Add chance level line at 25%
- [ ] **12.12** Add percentage text labels above bars
- [ ] **12.13** Use mode-specific colors from `COLORS` dict
- [ ] **12.14** Save to `RESULTS_DIR/accuracy_bars.png`

### Cell 12d — Per-Subject Heatmap

- [ ] **12.15** Create seaborn heatmap of per-subject accuracy (%) with annotations
- [ ] **12.16** Use `YlOrRd` colormap with range [40, 100]
- [ ] **12.17** Save to `RESULTS_DIR/heatmap.png`

### Cell 12e — Paired Subject Delta Plot

- [ ] **12.18** Create 1×2 subplot figure
- [ ] **12.19** Subplot (a): scatter plot of baseline vs apacore accuracy per subject, with identity line and subject labels
- [ ] **12.20** Subplot (b): horizontal bar chart of per-subject Δ accuracy, colored by direction (positive = apacore color, negative = baseline color)
- [ ] **12.21** Save to `RESULTS_DIR/paired_delta.png`

### Cell 12f — Confusion Matrices

- [ ] **12.22** Create 1×2 subplot figure for baseline and apacore
- [ ] **12.23** Aggregate confusion matrices across all subjects and seeds for each mode
- [ ] **12.24** Normalize rows to percentages
- [ ] **12.25** Display as annotated heatmaps with class names
- [ ] **12.26** Add mode name and overall accuracy to subplot titles
- [ ] **12.27** Save to `RESULTS_DIR/confusion_matrices.png`

------

## Phase 13: Section 13 — Statistical Analysis

- [ ] **13.1** Implement `wilcoxon_test_report(all_results, model_name)`:
  - [ ] **13.1.1** For each subject: collect per-seed accuracies for both modes
  - [ ] **13.1.2** Include subject only if it has exactly `N_REQUIRED_SEEDS` results for both modes
  - [ ] **13.1.3** Report excluded subjects with reason (incomplete seeds)
  - [ ] **13.1.4** Compute paired differences (apacore mean - baseline mean)
  - [ ] **13.1.5** Count nonzero differences (effective N)
  - [ ] **13.1.6** Count and report zero differences
  - [ ] **13.1.7** Skip test if N_eff < 5 (report minimum achievable p-value)
  - [ ] **13.1.8** Run one-sided Wilcoxon (`alternative="greater"`, `zero_method="wilcox"`)
  - [ ] **13.1.9** Run two-sided Wilcoxon (`alternative="two-sided"`, `zero_method="wilcox"`)
  - [ ] **13.1.10** Print: H1, unit of analysis, statistic, one-sided p, two-sided p, significance marker (✓/✗ at α=0.05), mean Δ, per-subject Δ values
  - [ ] **13.1.11** Print warnings: reduced power if N_complete < 9, N_eff=5 caveat (min two-sided p = 0.0625)
- [ ] **13.2** Call `wilcoxon_test_report(ok_results, 'ATCNet')`
- [ ] **13.3** Verify test output is interpretable and correctly formatted

------

## Phase 14: Section 14 — Discussion, Export & Future Hooks

### Cell 14a — Markdown Discussion

- [ ] **14.1** Write baseline performance discussion: expected ~75–82%, comparison caveat with published ATCNet (1125 vs 1000 samples)
- [ ] **14.2** Write APA-Core effect discussion: explain each component’s contribution
- [ ] **14.3** Write limitations section: single model, no ablation executed, no DVA/LLM, MI window difference
- [ ] **14.4** Write future work section: additional models, ablation modes, DVA, LLM explainability, ML classifiers

### Cell 14b — Export

- [ ] **14.5** Save summary DataFrame as `summary_table.csv`
- [ ] **14.6** Save per-subject pivot table as `per_subject_accuracy.csv`
- [ ] **14.7** List all output files in `RESULTS_DIR` with sizes
- [ ] **14.8** Print notebook completion message

------

## Phase 15: Final Verification & Quality Assurance

- [ ] **15.1** Run the complete notebook end-to-end from a fresh Colab runtime (Runtime → Restart and run all) — smoke only, to verify no state leakage between cells
- [ ] **15.2** Verify all 18 data files are loaded without error
- [ ] **15.3** Verify smoke benchmark produces reasonable accuracy (~65–85% for S01 baseline)
- [ ] **15.4** Verify all plots are generated and saved correctly
- [ ] **15.5** Verify JSON output structure: envelope has `metadata`, `results`, `failures`, `last_updated`
- [ ] **15.6** Verify resume works: manually stop after a few runs, restart, resume from JSON, confirm it picks up where it left off and does not re-run completed runs
- [ ] **15.7** Verify resume validation catches mismatched config (e.g., change `t_start` and confirm `ValueError` is raised)
- [ ] **15.8** Verify atomic write works: confirm no partial/corrupted JSON files after interruption
- [ ] **15.9** Verify companion `.npz` file contains correct keys and array shapes
- [ ] **15.10** Verify `PROCEED` gate blocks full benchmark when smoke fails
- [ ] **15.11** Verify Wilcoxon test handles edge cases: all zero differences, N_eff < 5, incomplete subjects
- [ ] **15.12** Verify all plots use correct colors from `COLORS` dict
- [ ] **15.13** Verify all saved files are in `RESULTS_DIR` (no files written elsewhere)
- [ ] **15.14** Verify memory cleanup: `del` statements and `torch.cuda.empty_cache()` are correctly placed
- [ ] **15.15** Review all print statements for clarity and correct formatting
- [ ] **15.16** Verify cell execution order independence: each cell should work if run after all preceding cells, regardless of whether Section 11 full benchmark was actually executed
- [ ] **15.17** Final code review: check for any hardcoded values that should reference `Config`, any missing error handling, any TODO comments left in code

------

## Phase 16: Full Benchmark Execution & Results Collection

- [ ] **16.1** Set Colab to A100 runtime (if available)
- [ ] **16.2** Run Sections 1–10 to confirm smoke passes
- [ ] **16.3** Run Section 11 (full benchmark): monitor for first few runs, then allow to run unattended
- [ ] **16.4** If Colab disconnects: reconnect, mount Drive, re-run Sections 1–9 (definitions only), run Section 11b resume cell with correct `RESUME_PATH`
- [ ] **16.5** After full benchmark completes: run Sections 12–14 for visualization, statistics, and export
- [ ] **16.6** Review all output files: JSON, NPZ, CSVs, PNGs
- [ ] **16.7** Verify Wilcoxon p-value and interpret results
- [ ] **16.8** Archive final notebook and all results

------

**Total tasks: ~200 individual items across 16 phases**