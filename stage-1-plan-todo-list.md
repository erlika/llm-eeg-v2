Agreed. I’d **update the plan by making it explicitly staged**, not monolithic.

## What changes based on the latest comments

I agree with the reviewer on these points:

- **Do not merge everything into one giant patch.**
- **Run only the must-do fixes first**, then **stop at the pilot checkpoint**.
- Keep **strict-completeness paired analysis** as the **primary** analysis.
- Treat any later partial-seed sensitivity analysis as **exploratory only**.
- Extend preprocessing sanity checks to include a **test-shift check**.

So below is a **Stage 1 notebook patch pack** only.
 It includes exactly the must-do items:

1. fix normalization
2. add `strong_baseline`
3. make split fallback visible and recorded
4. add early stopping
5. run smoke + pilot
6. run **primary** paired report
7. enforce a **pilot decision checkpoint**

------

# How to apply this

## Important

- **Insert these new cells after Section 9** in your notebook.
- Then **use the new Stage 1 run cells below**.
- For now, **do not use the original Section 10/11 full-run flow** until the pilot checkpoint passes.

------

# STAGE 1 PATCH PACK

------

## Cell S1-A — Compatibility + updated config/modes

```python
# ==============================================================================
# STAGE 1 PATCH A: Compatibility + Config + Modes
# Insert after Section 9
# ==============================================================================

from dataclasses import dataclass, replace
from typing import Optional, Literal
import numpy as np
import pandas as pd

if 'COLORS' not in globals():
    COLORS = {
        'controlled_baseline': '#4C72B0',
        'strong_baseline': '#8172B2',
        'apacore': '#DD8452',
        'ablate': '#55A868',
    }
elif 'strong_baseline' not in COLORS:
    COLORS['strong_baseline'] = '#8172B2'

for _name in ['SUBJECTS', 'SEEDS', 'FS', 'N_CLASSES']:
    assert _name in globals(), f"Missing required global constant: {_name}"

@dataclass(frozen=True)
class Config:
    data_dir: str = 'data/'
    output_dir: str = 'results/'
    eval_labels_path: Optional[str] = None

    drop_artifacts: bool = True
    t_start: float = 2.0
    t_end: float = 6.0

    normalize: Literal[
        'none',
        'train_channel_global_zscore',
        'train_channel_timepoint_zscore',
    ] = 'train_channel_global_zscore'

    val_strategy: Literal['run_holdout', 'stratified'] = 'run_holdout'
    val_run_id: int = 6
    stratified_val_frac: float = 0.2

    ea_shrinkage: float = 0.01
    ea_eig_floor: float = 1e-6

    aug_noise_std_frac: float = 0.05
    aug_scale_range: tuple[float, float] = (0.9, 1.1)
    aug_jitter_max: int = 10

    lr_eta_min: float = 1e-6
    topk: int = 3
    min_epoch_gap: int = 5
    checkpoint_metric: Literal['accuracy', 'kappa'] = 'accuracy'

    seed: int = 42
    fail_fast: bool = False

    save_trial_outputs: bool = False
    save_training_history: bool = False

    # Updated default: pilot-friendly
    early_stopping_patience: Optional[int] = 50
    early_stopping_min_delta: float = 1e-4

    @property
    def n_times(self) -> int:
        return int((self.t_end - self.t_start) * FS)

    @property
    def mi_start_sample(self) -> int:
        return int(self.t_start * FS)

    @property
    def mi_end_sample(self) -> int:
        return int(self.t_end * FS)


@dataclass(frozen=True)
class ModeFlags:
    use_ea: bool
    use_aug: bool
    use_cosine: bool
    use_topk: bool


MODE_REGISTRY = {
    'controlled_baseline': ModeFlags(False, False, False, False),

    # NEW: fairer comparator
    'strong_baseline':     ModeFlags(False, False, True,  True),

    'apacore':             ModeFlags(True,  True,  True,  True),

    'ablate_no_ea':        ModeFlags(False, True,  True,  True),
    'ablate_no_aug':       ModeFlags(True,  False, True,  True),
    'ablate_no_cosine':    ModeFlags(True,  True,  False, True),
    'ablate_no_topk':      ModeFlags(True,  True,  True,  False),
}

_RESUME_EXEMPT_FIELDS = {
    'output_dir', 'fail_fast', 'seed',
    'save_training_history', 'save_trial_outputs'
}

CONFIG = Config(data_dir=DATA_DIR, output_dir=RESULTS_DIR)

print("Stage 1 patch A loaded.")
print(f"Config: n_times={CONFIG.n_times}, window=[{CONFIG.t_start}s, {CONFIG.t_end}s]")
print(f"Modes: {list(MODE_REGISTRY.keys())}")
```

------

## Cell S1-B — Split visibility + normalization patch + checkpoint helpers

```python
# ==============================================================================
# STAGE 1 PATCH B: Split + Normalization + Checkpoint helpers
# ==============================================================================

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.stratified_val_frac,
        random_state=config.seed,
    )
    train_idx, val_idx = next(splitter.split(X, y))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def smart_split(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: Optional[np.ndarray],
    config: Config,
    subject_id: Optional[int] = None,
):
    subject_tag = f"S{subject_id:02d}" if subject_id is not None else "unknown_subject"

    if config.val_strategy == 'run_holdout':
        if run_ids is not None and len(np.unique(run_ids)) > 1:
            mask_val = (run_ids == config.val_run_id)
            if int(mask_val.sum()) > 0:
                split_info = {
                    'requested_val_strategy': 'run_holdout',
                    'actual_val_strategy': 'run_holdout',
                    'val_run_id': int(config.val_run_id),
                    'split_fallback': False,
                    'n_val': int(mask_val.sum()),
                }
                return X[~mask_val], y[~mask_val], X[mask_val], y[mask_val], split_info

        print(
            f"[WARN] {subject_tag}: requested run_holdout with val_run_id={config.val_run_id}, "
            f"but that run was unavailable. Falling back to stratified split."
        )
        X_train, y_train, X_val, y_val = _stratified_split(X, y, config)
        split_info = {
            'requested_val_strategy': 'run_holdout',
            'actual_val_strategy': 'stratified',
            'val_run_id': None,
            'split_fallback': True,
            'n_val': int(len(y_val)),
        }
        return X_train, y_train, X_val, y_val, split_info

    if config.val_strategy == 'stratified':
        X_train, y_train, X_val, y_val = _stratified_split(X, y, config)
        split_info = {
            'requested_val_strategy': 'stratified',
            'actual_val_strategy': 'stratified',
            'val_run_id': None,
            'split_fallback': False,
            'n_val': int(len(y_val)),
        }
        return X_train, y_train, X_val, y_val, split_info

    raise ValueError(f"Unknown val_strategy: {config.val_strategy}")


def _apply_global_channel_zscore(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    eps: float = 1e-8,  # dead-channel safeguard
):
    X_tr = X_train.copy().astype(np.float32)
    X_va = None if X_val is None else X_val.copy().astype(np.float32)
    X_te = X_test.copy().astype(np.float32)

    n_channels = X_tr.shape[1]
    for ch in range(n_channels):
        mean = float(X_tr[:, ch, :].mean())
        std = float(X_tr[:, ch, :].std())
        std = max(std, eps)

        X_tr[:, ch, :] = (X_tr[:, ch, :] - mean) / std
        if X_va is not None:
            X_va[:, ch, :] = (X_va[:, ch, :] - mean) / std
        X_te[:, ch, :] = (X_te[:, ch, :] - mean) / std

    return X_tr, X_va, X_te


def _apply_timepoint_channel_zscore(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
):
    X_tr = X_train.copy()
    X_va = None if X_val is None else X_val.copy()
    X_te = X_test.copy()

    n_channels = X_tr.shape[1]
    for ch in range(n_channels):
        scaler = StandardScaler()
        scaler.fit(X_tr[:, ch, :])  # old behavior kept only for A/B testing
        X_tr[:, ch, :] = scaler.transform(X_tr[:, ch, :])
        if X_va is not None:
            X_va[:, ch, :] = scaler.transform(X_va[:, ch, :])
        X_te[:, ch, :] = scaler.transform(X_te[:, ch, :])

    return (
        X_tr.astype(np.float32),
        None if X_va is None else X_va.astype(np.float32),
        X_te.astype(np.float32),
    )


def apply_normalization(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    config: Config,
    run_label: Optional[str] = None,
):
    prefix = f"[{run_label}] " if run_label else ""
    print(f"{prefix}Normalization: {config.normalize}")

    if config.normalize == 'none':
        return (
            X_train.astype(np.float32),
            None if X_val is None else X_val.astype(np.float32),
            X_test.astype(np.float32),
        )

    if config.normalize == 'train_channel_global_zscore':
        return _apply_global_channel_zscore(X_train, X_val, X_test)

    if config.normalize == 'train_channel_timepoint_zscore':
        return _apply_timepoint_channel_zscore(X_train, X_val, X_test)

    raise ValueError(f"Unknown normalization mode: {config.normalize}")


# Add small checkpoint diagnostics without rewriting the whole class
def _ckpt_epochs(self):
    return [int(entry.epoch) for entry in self.entries]

def _ckpt_metrics(self):
    return [float(entry.val_metric) for entry in self.entries]

CheckpointManager.checkpoint_epochs = _ckpt_epochs
CheckpointManager.checkpoint_metrics = _ckpt_metrics

print("Stage 1 patch B loaded.")
```

------

## Cell S1-C — Redefine `_prepare_data`, `_train_model`, and `run_single`

```python
# ==============================================================================
# STAGE 1 PATCH C: _prepare_data + _train_model + run_single
# ==============================================================================

def _prepare_data(subject_id: int, mode_flags: ModeFlags, config: Config):
    X_T, y_T, run_ids, _ = load_bci2a_session(subject_id, 'T', config)
    X_E, y_E, _, _ = load_bci2a_session(subject_id, 'E', config)

    X_train, y_train, X_val, y_val, split_info = smart_split(
        X_T, y_T, run_ids, config, subject_id=subject_id
    )
    X_test, y_test = X_E, y_E

    if mode_flags.use_ea:
        ea = DiagonalShrinkageEA(config.ea_shrinkage, config.ea_eig_floor)
        ea.fit(X_train)
        X_train = ea.transform(X_train)
        X_val = ea.transform(X_val)
        X_test = ea.transform(X_test)

    run_label = f"S{subject_id:02d}"
    X_train, X_val, X_test = apply_normalization(
        X_train, X_val, X_test, config, run_label=run_label
    )

    print(
        f"[{run_label}] Split: requested={split_info['requested_val_strategy']} "
        f"actual={split_info['actual_val_strategy']} "
        f"n_train={len(y_train)} n_val={len(y_val)} n_test={len(y_test)}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, split_info


def _train_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mode_flags: ModeFlags,
    config: Config,
    device: torch.device,
):
    params = spec.default_train_params()
    n_channels, n_times = X_train.shape[1], X_train.shape[2]

    model = spec.build_model(
        n_classes=N_CLASSES,
        n_channels=n_channels,
        n_times=n_times
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params['lr']),
        weight_decay=float(params['weight_decay'])
    )

    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=int(params['epochs']),
            eta_min=config.lr_eta_min
        ) if mode_flags.use_cosine else None
    )

    criterion = nn.CrossEntropyLoss()
    augmenter = (
        MildAugmenter(
            config.aug_noise_std_frac,
            config.aug_scale_range,
            config.aug_jitter_max
        ) if mode_flags.use_aug else None
    )
    ckpt = CheckpointManager(config.topk, config.min_epoch_gap)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=int(params['batch_size']),
        shuffle=True,
        generator=generator,
        worker_init_fn=_seed_worker,
    )

    train_start = time.time()
    history = {'train_loss': [], 'val_metric': [], 'lr': []}

    best_metric = -np.inf
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(int(params['epochs'])):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if augmenter is not None:
                xb = augmenter(xb)

            xb = spec.reshape_input(xb)
            logits = spec.extract_logits(model(xb))
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        val_probs = _batched_predict(
            model,
            X_val_t,
            device,
            batch_size=int(params['batch_size']),
            reshape_fn=spec.reshape_input,
            extract_fn=spec.extract_logits,
        )
        val_preds = np.argmax(val_probs, axis=1)

        val_metric = (
            cohen_kappa(y_val, val_preds)
            if config.checkpoint_metric == 'kappa'
            else accuracy(y_val, val_preds)
        )

        ckpt.update(epoch, val_metric, model)

        current_lr = optimizer.param_groups[0]['lr']
        if config.save_training_history:
            history['train_loss'].append(epoch_loss / max(n_batches, 1))
            history['val_metric'].append(val_metric)
            history['lr'].append(current_lr)

        if val_metric > best_metric + config.early_stopping_min_delta:
            best_metric = val_metric
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (
            config.early_stopping_patience is not None
            and epochs_no_improve >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(best_epoch={best_epoch + 1}, best_val={best_metric:.4f})"
            )
            break

    train_time = time.time() - train_start
    history['best_epoch'] = int(best_epoch)
    history['best_val_metric'] = float(best_metric)
    history['checkpoint_epochs'] = ckpt.checkpoint_epochs()
    history['checkpoint_metrics'] = ckpt.checkpoint_metrics()
    history['stopped_early'] = bool(stopped_early)
    return model, ckpt, train_time, history


def run_single(spec: ModelSpec, subject_id: int, mode_name: str, config: Config, device: torch.device):
    mode_flags = MODE_REGISTRY[mode_name]
    _seed_everything(config.seed)

    X_tr, y_tr, X_va, y_va, X_te, y_te, split_info = _prepare_data(
        subject_id, mode_flags, config
    )
    model, ckpt, train_time, history = _train_model(
        spec, X_tr, y_tr, X_va, y_va, mode_flags, config, device
    )
    eval_info = _evaluate_model(spec, model, ckpt, X_te, y_te, mode_flags, config, device)

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

        # metadata for reproducibility / debugging
        'normalize': config.normalize,
        'requested_val_strategy': split_info['requested_val_strategy'],
        'actual_val_strategy': split_info['actual_val_strategy'],
        'split_fallback': split_info['split_fallback'],
        'val_run_id': split_info['val_run_id'],

        # training diagnostics
        'best_epoch': history.get('best_epoch'),
        'best_val_metric': history.get('best_val_metric'),
        'checkpoint_epochs': history.get('checkpoint_epochs', []),
        'checkpoint_metrics': history.get('checkpoint_metrics', []),
        'stopped_early': history.get('stopped_early', False),

        'status': 'ok',
        'error': None,
    }

    trial_outputs = None
    if config.save_trial_outputs:
        trial_outputs = {
            'y_true': eval_info['y_true'],
            'y_pred': eval_info['y_pred'],
            'y_proba': eval_info['y_proba'],
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, trial_outputs, history

print("Stage 1 patch C loaded.")
```

------

## Cell S1-D — Preprocessing sanity + primary paired report + pilot gate

```python
# ==============================================================================
# STAGE 1 PATCH D: Sanity checks + PRIMARY stats + Pilot checkpoint
# ==============================================================================

from scipy.stats import wilcoxon, rankdata, binomtest

def preprocessing_sanity(subject_id: int, mode_name: str, config: Config):
    mode_flags = MODE_REGISTRY[mode_name]
    X_tr, y_tr, X_va, y_va, X_te, y_te, split_info = _prepare_data(subject_id, mode_flags, config)

    ch_mean = X_tr.mean(axis=(0, 2))
    ch_std = X_tr.std(axis=(0, 2))

    train_abs_max = float(np.max(np.abs(X_tr)))
    test_abs_max = float(np.max(np.abs(X_te)))
    train_q99 = float(np.percentile(np.abs(X_tr), 99))
    test_q99 = float(np.percentile(np.abs(X_te), 99))
    q99_ratio = test_q99 / (train_q99 + 1e-8)

    print(f"\nSubject S{subject_id:02d} | mode={mode_name}")
    print(f"Train shape: {X_tr.shape} | Val shape: {X_va.shape} | Test shape: {X_te.shape}")
    print(f"Requested split: {split_info['requested_val_strategy']}")
    print(f"Actual split:    {split_info['actual_val_strategy']}")
    print(f"Mean |mean| across channels: {np.mean(np.abs(ch_mean)):.6f}")
    print(f"Mean std across channels:    {np.mean(ch_std):.6f}")
    print(f"Min/max channel mean: [{ch_mean.min():.6f}, {ch_mean.max():.6f}]")
    print(f"Min/max channel std:  [{ch_std.min():.6f}, {ch_std.max():.6f}]")
    print(f"Train abs max: {train_abs_max:.4f}")
    print(f"Test  abs max: {test_abs_max:.4f}")
    print(f"Train |x| 99th pct: {train_q99:.4f}")
    print(f"Test  |x| 99th pct: {test_q99:.4f}")
    print(f"Test/Train 99th pct ratio: {q99_ratio:.3f}")

    if np.mean(np.abs(ch_mean)) > 1e-2:
        print("[WARN] Channel means are not close to zero.")
    if not (0.9 <= np.mean(ch_std) <= 1.1):
        print("[WARN] Average channel std is not close to one.")
    if q99_ratio > 3.0:
        print("[WARN] Test distribution looks much larger than train after normalization; possible domain shift.")
    if split_info['split_fallback']:
        print("[WARN] Validation strategy fell back from run_holdout to stratified.")


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = 20000, seed: int = 42, ci: float = 95.0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    n = len(diff)

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()

    alpha = (100.0 - ci) / 2.0
    return np.percentile(boot, [alpha, 100.0 - alpha])


def signed_rank_rank_biserial(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method='average')
    w_plus = ranks[diff > 0].sum()
    w_minus = ranks[diff < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus + 1e-12))


def subject_mean_table_primary(
    all_results: list[dict[str, object]],
    left_mode: str,
    right_mode: str,
    required_seeds: int,
):
    rows = []
    excluded = []

    for subj in SUBJECTS:
        left = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == left_mode and r['subject'] == subj
        ]
        right = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == right_mode and r['subject'] == subj
        ]

        if len(left) == required_seeds and len(right) == required_seeds:
            rows.append({
                'subject': subj,
                'left_mean': np.mean(left),
                'right_mean': np.mean(right),
                'delta': np.mean(right) - np.mean(left),
                'left_std_seed': np.std(left),
                'right_std_seed': np.std(right),
                'left_n_seed': len(left),
                'right_n_seed': len(right),
            })
        else:
            excluded.append({
                'subject': subj,
                'left_n_seed': len(left),
                'right_n_seed': len(right),
            })

    return pd.DataFrame(rows), excluded


def paired_mode_report_primary(
    all_results: list[dict[str, object]],
    left_mode: str,
    right_mode: str,
    required_seeds: int,
    directional_hypothesis_pre_specified: bool = False,
):
    print("=" * 70)
    print(f"PRIMARY analysis (strict completeness): {right_mode} vs {left_mode}")
    print("=" * 70)

    df_pair, excluded = subject_mean_table_primary(
        all_results, left_mode, right_mode, required_seeds=required_seeds
    )

    if excluded:
        print("Excluded subjects:")
        for e in excluded:
            print(f"  S{e['subject']:02d}: left={e['left_n_seed']} right={e['right_n_seed']}")

    if len(df_pair) == 0:
        print("No complete paired subjects available.")
        return None

    print(df_pair[['subject', 'left_mean', 'right_mean', 'delta']].round(4).to_string(index=False))

    diff = df_pair['delta'].to_numpy()
    n_positive = int(np.sum(diff > 0))
    n_negative = int(np.sum(diff < 0))
    n_nonzero = int(np.count_nonzero(diff))

    print("-" * 70)
    print(f"N paired subjects: {len(df_pair)}")
    print(f"Improved / worsened / tied: {n_positive} / {n_negative} / {len(df_pair) - n_positive - n_negative}")
    print(f"Mean delta:   {diff.mean():.4f}")
    print(f"Median delta: {np.median(diff):.4f}")

    ci_low, ci_high = bootstrap_mean_ci(diff)
    print(f"Bootstrap 95% CI for mean delta: [{ci_low:.4f}, {ci_high:.4f}]")

    if n_positive + n_negative > 0:
        sign_alt = 'greater' if directional_hypothesis_pre_specified else 'two-sided'
        sign_p = binomtest(
            n_positive,
            n_positive + n_negative,
            p=0.5,
            alternative=sign_alt
        ).pvalue
        print(f"Sign test p ({sign_alt}): {sign_p:.4f}")

    if n_nonzero >= 5:
        stat_two, p_two = wilcoxon(
            df_pair['right_mean'].to_numpy(),
            df_pair['left_mean'].to_numpy(),
            alternative='two-sided',
            zero_method='wilcox'
        )
        print(f"Wilcoxon statistic: {stat_two:.4f}")
        print(f"Wilcoxon two-sided p: {p_two:.4f}")

        if directional_hypothesis_pre_specified:
            _, p_one = wilcoxon(
                df_pair['right_mean'].to_numpy(),
                df_pair['left_mean'].to_numpy(),
                alternative='greater',
                zero_method='wilcox'
            )
            print(f"Wilcoxon one-sided p (pre-specified): {p_one:.4f}")

        print(f"Rank-biserial effect size: {signed_rank_rank_biserial(diff):.4f}")
    else:
        print("Wilcoxon skipped: too few nonzero paired differences.")

    print("=" * 70)
    return df_pair


def pilot_decision_checkpoint(
    all_results: list[dict[str, object]],
    left_mode: str = 'strong_baseline',
    right_mode: str = 'apacore',
    required_seeds: int = 2,
    mean_delta_threshold_pp: float = 1.0,
):
    df_pair, excluded = subject_mean_table_primary(
        all_results,
        left_mode=left_mode,
        right_mode=right_mode,
        required_seeds=required_seeds,
    )

    if excluded:
        print("Pilot checkpoint excluded subjects:")
        for e in excluded:
            print(f"  S{e['subject']:02d}: left={e['left_n_seed']} right={e['right_n_seed']}")

    if len(df_pair) == 0:
        print("No complete pilot pairs. Do NOT proceed.")
        return False

    diff_pp = df_pair['delta'].to_numpy() * 100
    n_improved = int(np.sum(diff_pp > 0))
    n_worsened = int(np.sum(diff_pp < 0))
    mean_delta_pp = float(diff_pp.mean())

    print("=" * 70)
    print("Pilot decision checkpoint")
    print("=" * 70)
    print(f"Mean delta: {mean_delta_pp:.2f} pp")
    print(f"Improved / worsened / tied: {n_improved} / {n_worsened} / {len(diff_pp) - n_improved - n_worsened}")

    if mean_delta_pp < 0:
        print("Recommendation: STOP and investigate. APA-Core is negative in pilot.")
        return False

    if abs(mean_delta_pp) < mean_delta_threshold_pp and n_improved <= n_worsened:
        print("Recommendation: HOLD. Effect is tiny/inconsistent; inspect normalization, EA, and augmentation first.")
        return False

    print("Recommendation: OK to proceed to the next stage.")
    return True

print("Stage 1 patch D loaded.")
```

------

## Cell S1-E — Stage 1 config

This is the only config you should use right now.

```python
# ==============================================================================
# STAGE 1 PATCH E: Fair APA config only
# ==============================================================================

FAIR_APA_CONFIG = replace(
    CONFIG,
    t_start=2.0,
    t_end=6.0,
    normalize='train_channel_global_zscore',
    val_strategy='run_holdout',
    early_stopping_patience=50,
    save_trial_outputs=True,
)

print(f"FAIR_APA_CONFIG n_times={FAIR_APA_CONFIG.n_times}")
```

------

# STAGE 1 RUN CELLS

------

## Cell S1-R1 — Preprocessing sanity check

Run this before smoke.

```python
# ==============================================================================
# STAGE 1 RUN 1: Preprocessing sanity check
# ==============================================================================

preprocessing_sanity(
    subject_id=1,
    mode_name='strong_baseline',
    config=FAIR_APA_CONFIG,
)
```

### What you want to see

- channel means near 0
- channel std near 1
- no split fallback unless expected
- test/train 99th percentile ratio not absurdly huge

------

## Cell S1-R2 — Smoke benchmark

Use this instead of the original smoke cell.

```python
# ==============================================================================
# STAGE 1 RUN 2: Smoke benchmark
# ==============================================================================

spec = MODEL_REGISTRY['atcnet']

smoke_results, smoke_failures, smoke_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=replace(FAIR_APA_CONFIG, save_training_history=True),
    modes=['controlled_baseline', 'strong_baseline', 'apacore'],
    subjects=[1],
    seeds=[42],
    tag='smoke_stage1',
)

print("\nSmoke results:")
for r in smoke_results:
    print(
        f"{r['mode']:20s} | acc={r['accuracy']:.4f} | "
        f"kappa={r['kappa']:.4f} | best_epoch={r.get('best_epoch')} | "
        f"fallback={r.get('split_fallback')}"
    )

if smoke_failures:
    print("\nSmoke failures:")
    for f in smoke_failures:
        print(f)
```

### Immediate question after smoke

You want to see whether:

- `strong_baseline` > `controlled_baseline`
- normalization/logging/splits look sane
- training stops at a reasonable epoch

------

## Cell S1-R3 — Pilot benchmark

This is the gate before any full run.

```python
# ==============================================================================
# STAGE 1 RUN 3: Pilot benchmark
# ==============================================================================

pilot_results, pilot_failures, pilot_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=FAIR_APA_CONFIG,
    modes=[
        'controlled_baseline',
        'strong_baseline',
        'apacore',
        'ablate_no_ea',
        'ablate_no_aug',
    ],
    subjects=[1, 2, 3],
    seeds=[42, 123],
    tag='pilot_stage1',
)

print(f"\nPilot saved to: {pilot_path}")
print(f"Pilot failures: {len(pilot_failures)}")
```

------

## Cell S1-R4 — Primary paired report for the pilot

Use strict completeness only.

```python
# ==============================================================================
# STAGE 1 RUN 4: PRIMARY paired report (strict completeness)
# ==============================================================================

pilot_pair_df = paired_mode_report_primary(
    pilot_results,
    left_mode='strong_baseline',
    right_mode='apacore',
    required_seeds=2,
    directional_hypothesis_pre_specified=False,  # safer default
)
```

------

## Cell S1-R5 — Pilot decision checkpoint

This is the stop/go cell.

```python
# ==============================================================================
# STAGE 1 RUN 5: Pilot decision checkpoint
# ==============================================================================

pilot_ok = pilot_decision_checkpoint(
    pilot_results,
    left_mode='strong_baseline',
    right_mode='apacore',
    required_seeds=2,
    mean_delta_threshold_pp=1.0,
)

print(f"\npilot_ok = {pilot_ok}")
```

------

# What to do after Stage 1

## If `pilot_ok == False`

**Stop. Do not run a full benchmark yet.**

Investigate:

- whether `strong_baseline` improved over `controlled_baseline`
- whether `ablate_no_ea` beats `apacore`
- whether `ablate_no_aug` beats `apacore`
- whether test/train shift looks large in the sanity check
- whether checkpoint epochs cluster too tightly

In that case, the next step is **not** a full run — it’s parameter adjustment.

------

## If `pilot_ok == True`

Then you move to **Stage 2**, which should include:

1. reproduction-oriented check with `[1.5, 6.0]`
2. full fair comparison: `strong_baseline` vs `apacore`
3. ablations
4. optional sensitivity analysis for incomplete runs

But per the reviewer’s point, I **would not preload all of that now**.

------

# Important methodological note

When we later add the partial-seed sensitivity analysis, I would put this comment directly in the code:

> **If primary and sensitivity analyses disagree, trust the primary strict-completeness result.**

That keeps the discipline the reviewer is asking for.

------

# My updated recommendation in one sentence

Use this **Stage 1 patch only**, run the pilot, and **actually honor the checkpoint** before expanding to reproduction checks or full ablations.

If you want, I can next give you the **Stage 2 patch pack** for:

- reproduction config,
- full benchmark cells,
- ablation reporting,
- exploratory sensitivity analysis.



-----

Absolutely — here’s a **detailed, staged TODO list** you can append to the plan so it’s operational, trackable, and aligned with the reviewer’s comments.

------

# Detailed TODO List for the Staged Plan

## Phase 0 — Freeze scope and guardrails

**Goal:** lock the process so Stage 1 stays minimal and decision-driven.

- [x] Confirm that **Stage 1 is the only active implementation scope** for now.
- [x] Explicitly **defer** any full benchmark, reproduction sweep, or sensitivity analysis until after the pilot checkpoint.
- [ ] Record the analysis policy:
  - [x] **Primary analysis** = strict-completeness paired analysis.
  - [x] **Sensitivity analysis** = optional, exploratory, secondary only.
- [ ] Record the execution policy:
  - [x] **Do not use the old Section 10/11 full-run flow** until the pilot checkpoint passes.
  - [x] **Do not merge future Stage 2 work into the Stage 1 patch.**
- [x] Create a notebook backup / git commit before patching.
- [x] Create a results/log directory for Stage 1 outputs if it does not already exist.

**Phase 0 done when:**

- scope is frozen,
- backup exists,
- everyone agrees that Stage 1 ends at the pilot checkpoint.

------

## Phase 1 — Integrate the Stage 1 patch pack

**Goal:** insert only the must-do notebook modifications.

### 1.1 Insert patch cells

- [x] Insert **Cell S1-A** after Section 9.
- [x] Insert **Cell S1-B** after Section 9.
- [x] Insert **Cell S1-C** after Section 9.
- [x] Insert **Cell S1-D** after Section 9.
- [x] Insert **Cell S1-E** after Section 9.

### 1.2 Verify compatibility assumptions

- [ ] Confirm required globals exist before running:
  - [x] `SUBJECTS`
  - [x] `SEEDS`
  - [x] `FS`
  - [x] `N_CLASSES`
  - [x] `DATA_DIR`
  - [x] `RESULTS_DIR`
- [ ] Confirm existing notebook definitions are present and compatible:
  - [x] `CheckpointManager`
  - [x] `ModelSpec`
  - [x] `MODEL_REGISTRY`
  - [x] `load_bci2a_session`
  - [x] `DiagonalShrinkageEA`
  - [x] `MildAugmenter`
  - [x] `_batched_predict`
  - [x] `_evaluate_model`
  - [x] `run_full_benchmark`
  - [x] `_seed_everything`
  - [x] `_seed_worker`
- [x] Confirm `torch`, `numpy`, `pandas`, `scipy`, and `sklearn` imports are available in the notebook runtime.

### 1.3 Confirm Stage 1 config is active

- [x] Run **Cell S1-E** (added to notebook).
- [ ] Verify:
  - [x] `t_start == 2.0`
  - [x] `t_end == 6.0`
  - [x] `normalize == 'train_channel_global_zscore'`
  - [x] `val_strategy == 'run_holdout'`
  - [x] `early_stopping_patience == 50`
  - [x] `save_trial_outputs == True`

**Phase 1 done when:**

- all Stage 1 cells execute cleanly,
- compatibility assumptions are satisfied,
- `FAIR_APA_CONFIG` is the active config.

------

## Phase 2 — Validate the implementation changes

**Goal:** verify that the must-do fixes are really present and observable.

### 2.1 Normalization fix

- [x] Confirm the active normalization mode is `train_channel_global_zscore`.
- [x] Confirm normalization is fit from **training only** and applied to validation/test.
- [x] Confirm dead-channel safeguard (`eps`) is active.
- [x] Keep `train_channel_timepoint_zscore` available only as a legacy A/B option, not the default.

### 2.2 Strong baseline added

- [x] Confirm `MODE_REGISTRY` includes `strong_baseline`.
- [ ] Confirm `strong_baseline` uses:
  - [x] no EA
  - [x] no augmentation
  - [x] cosine scheduler
  - [x] top-k checkpointing

### 2.3 Split fallback visibility

- [x] Confirm `smart_split()` returns `split_info`.
- [x] Confirm warnings are printed if run-holdout is unavailable.
- [ ] Confirm output results record:
  - [x] `requested_val_strategy`
  - [x] `actual_val_strategy`
  - [x] `split_fallback`
  - [x] `val_run_id`

### 2.4 Early stopping

- [x] Confirm early stopping logic is active in `_train_model()`.
- [ ] Confirm the stopping criteria use:
  - [x] `early_stopping_patience`
  - [x] `early_stopping_min_delta`
- [ ] Confirm training history records:
  - [x] `best_epoch`
  - [x] `best_val_metric`
  - [x] `stopped_early`
  - [x] `checkpoint_epochs`
  - [x] `checkpoint_metrics`

### 2.5 Test-shift sanity check

- [x] Confirm `preprocessing_sanity()` reports:
  - [x] train/val/test shapes
  - [x] split strategy
  - [x] channel means/stds
  - [x] train/test max magnitude
  - [x] train/test 99th percentile magnitude
  - [x] test/train 99th-percentile ratio
- [x] Confirm it warns on suspicious test-shift.

**Phase 2 done when:**

- each must-do fix is visibly implemented,
- the notebook prints or records the expected diagnostics.

------

## Phase 3 — Run the preprocessing sanity check

**Goal:** ensure normalization and split logic are sane before training anything substantial.

### 3.1 Execute sanity check

- [ ] Run **Cell S1-R1** with:
  - [ ] `subject_id=1`
  - [ ] `mode_name='strong_baseline'`
  - [ ] `config=FAIR_APA_CONFIG`

### 3.2 Review sanity outputs

- [ ] Check channel means are approximately zero.
- [ ] Check average channel std is approximately one.
- [ ] Check no unexpected split fallback occurred.
- [ ] Check the test/train 99th percentile ratio is not implausibly large.
- [ ] Note whether any domain-shift warning is printed.

### 3.3 Log findings

- [ ] Save the sanity output to notes / notebook markdown / experiment log.
- [ ] Record whether Stage 1 can proceed without patch changes.

**Stop condition:**

- [ ] If sanity check looks broken, **stop here** and debug preprocessing before smoke.

**Phase 3 done when:**

- sanity output is reviewed and explicitly marked pass/fail.

------

## Phase 4 — Run the smoke benchmark

**Goal:** do a minimal end-to-end run to verify training/evaluation behavior.

### 4.1 Execute smoke run

- [ ] Run **Cell S1-R2** using:
  - [ ] `spec = MODEL_REGISTRY['atcnet']`
  - [ ] modes = `['controlled_baseline', 'strong_baseline', 'apacore']`
  - [ ] subjects = `[1]`
  - [ ] seeds = `[42]`

### 4.2 Review smoke outputs

- [ ] Confirm all three modes finish successfully.
- [ ] Confirm no unexpected crashes/failures.
- [ ] Check whether `strong_baseline` improves over `controlled_baseline`.
- [ ] Check whether training stops at a plausible epoch.
- [ ] Check fallback metadata is printed and sensible.
- [ ] Check saved histories exist if enabled.

### 4.3 Diagnose smoke issues if present

- [ ] If `strong_baseline` is worse than `controlled_baseline`, flag this for investigation.
- [ ] If `apacore` is much worse than expected, flag this for investigation.
- [ ] If stopping happens implausibly early or too late, flag patience/metric tuning.
- [ ] If split fallback happened unexpectedly, inspect `run_ids` and subject/session handling.

**Stop condition:**

- [ ] If smoke behavior is clearly broken, **do not proceed to pilot**.

**Phase 4 done when:**

- smoke run completes,
- outputs are reviewed,
- issues are either absent or explicitly triaged.

------

## Phase 5 — Run the pilot benchmark

**Goal:** collect the minimum evidence needed for the checkpoint decision.

### 5.1 Execute pilot run

- [ ] Run **Cell S1-R3** with:
  - [ ] modes:
    - [ ] `controlled_baseline`
    - [ ] `strong_baseline`
    - [ ] `apacore`
    - [ ] `ablate_no_ea`
    - [ ] `ablate_no_aug`
  - [ ] subjects = `[1, 2, 3]`
  - [ ] seeds = `[42, 123]`

### 5.2 Verify pilot completeness

- [ ] Confirm pilot results are saved.
- [ ] Confirm failure count is recorded.
- [ ] Check that each subject/mode has the expected two seeds wherever possible.
- [ ] Identify any incomplete subject-mode pairs.
- [ ] Record any failures separately with traceback/error context.

### 5.3 Quick pilot QA

- [ ] Inspect per-run metadata for:
  - [ ] normalization mode
  - [x] split strategy
  - [ ] split fallback
  - [ ] best epoch
  - [ ] stopped early
- [ ] Confirm there is no systematic split fallback issue.
- [ ] Confirm early stopping does not collapse nearly all runs to the same tiny epoch count.

**Phase 5 done when:**

- pilot is complete enough for the primary paired report,
- failures and incompleteness are documented.

------

## Phase 6 — Run the primary paired analysis

**Goal:** perform the main reviewer-aligned pilot comparison.

### 6.1 Execute strict-completeness paired report

- [ ] Run **Cell S1-R4**:
  - [ ] `left_mode='strong_baseline'`
  - [ ] `right_mode='apacore'`
  - [ ] `required_seeds=2`
  - [ ] `directional_hypothesis_pre_specified=False`

### 6.2 Review primary analysis outputs

- [ ] Confirm excluded subjects are listed.
- [ ] Confirm only subjects with complete seed coverage are included.
- [ ] Review:
  - [ ] subject-wise means
  - [ ] subject-wise deltas
  - [ ] mean delta
  - [ ] median delta
  - [ ] bootstrap CI
  - [ ] sign test
  - [ ] Wilcoxon result if applicable
  - [ ] rank-biserial effect size

### 6.3 Record interpretation

- [ ] Write a brief summary of whether `apacore` appears better, tied, or worse than `strong_baseline`.
- [ ] Explicitly note that this is the **primary analysis**.
- [ ] Explicitly note that no incomplete-pair sensitivity analysis has been used for decision-making.

**Phase 6 done when:**

- a strict-completeness primary report exists and is interpreted.

------

## Phase 7 — Enforce the pilot decision checkpoint

**Goal:** make a real stop/go decision before any expansion.

### 7.1 Execute checkpoint

- [ ] Run **Cell S1-R5**:
  - [ ] `left_mode='strong_baseline'`
  - [ ] `right_mode='apacore'`
  - [ ] `required_seeds=2`
  - [ ] `mean_delta_threshold_pp=1.0`

### 7.2 Record checkpoint result

- [ ] Save the boolean `pilot_ok`.
- [ ] Record the checkpoint summary:
  - [ ] mean delta in percentage points
  - [ ] improved / worsened / tied counts
  - [ ] recommendation text

### 7.3 Enforce branching rule

- [ ] If `pilot_ok == False`, **stop** and move to the diagnosis/tuning phase.
- [ ] If `pilot_ok == True`, unlock Stage 2 planning and execution.

**Phase 7 done when:**

- there is an explicit stop/go result,
- the next step follows that result rather than bypassing it.

------

# Branch A — If `pilot_ok == False`

## Phase 8A — Diagnose before expanding

**Goal:** investigate weaknesses instead of launching the full benchmark.

### 8A.1 Compare baselines

- [ ] Check whether `strong_baseline` improved over `controlled_baseline`.
- [ ] If not, determine whether scheduler/top-k changes are underperforming or unstable.

### 8A.2 Inspect ablations

- [ ] Compare `ablate_no_ea` vs `apacore`.
- [ ] Compare `ablate_no_aug` vs `apacore`.
- [ ] Determine whether EA or augmentation is hurting pilot performance.

### 8A.3 Inspect preprocessing/test-shift

- [ ] Revisit sanity outputs for train/test scale mismatch.
- [ ] Check whether any subject has unusually large test/train q99 ratio.
- [ ] Check whether normalization behavior differs across subjects.

### 8A.4 Inspect training behavior

- [ ] Review `best_epoch`, `checkpoint_epochs`, and `checkpoint_metrics`.
- [ ] Check whether checkpoints cluster too tightly.
- [ ] Check whether early stopping patience is too short or too permissive.

### 8A.5 Decide next adjustment loop

- [ ] Select the smallest justified parameter/config change.
- [ ] Re-run sanity + smoke after any change.
- [ ] Re-run the pilot only after the updated configuration looks defensible.

### 8A.6 Documentation

- [ ] Document why the pilot failed or held.
- [ ] Document what changes will be tested next.
- [ ] Keep Stage 2 locked until a revised pilot passes.

**Phase 8A done when:**

- a concrete diagnosis exists,
- the next parameter-adjustment cycle is defined.

------

# Branch B — If `pilot_ok == True`

## Phase 8B — Prepare Stage 2 expansion

**Goal:** only after passing pilot, expand to broader evaluation.

### 8B.1 Freeze Stage 1 outputs

- [x] Save final Stage 1 results tables. *(S2-R1: `freeze_stage_artifacts` saves pilot manifest)*
- [x] Save pilot logs and any per-trial outputs. *(S2-R1: freeze manifest includes coverage + config)*
- [x] Record the exact config used in the successful pilot. *(FAIR_APA_CONFIG captured in freeze manifest)*
- [x] Commit notebook/code state before Stage 2 changes. *(commit `21c4ef8` fold_id fix; patch_stage2.py inserted 21 cells)*

### 8B.2 Define Stage 2 scope

- [x] Reproduction-oriented check with window `[1.5, 6.0]` *(REPRO_1506_CONFIG in S2-E; S2-R9)*
- [x] Full fair comparison: `strong_baseline` vs `apacore` *(STAGE2_FAIR_MODES in S2-E; S2-R4, S2-R5)*
- [x] Ablation study *(STAGE2_ABLATION_MODES in S2-E; S2-R6, S2-R7)*
- [x] Optional exploratory incomplete-run sensitivity analysis *(S2-R11)*

**Phase 8B done when:**

- Stage 1 is archived, ✅
- Stage 2 scope is explicitly approved. ✅ (stage-2-plan-todo-list-w-code.md)

------

## Phase 9 — Stage 2 execution

**Goal:** run the broader evaluation only after passing the pilot gate.

### 9.1 Reproduction-oriented check

- [x] Create a reproduction config using `[1.5, 6.0]`. *(REPRO_1506_CONFIG = replace(FAIR_FULL_CONFIG, t_start=1.5, t_end=6.0))*
- [x] Run a targeted reproduction-oriented benchmark. *(S2-R9: `run_full_benchmark` with REPRO_1506_CONFIG)*
- [x] Compare results against Stage 1 fair config outcomes. *(S2-R10: paired_mode_report_primary)*
- [x] Label this analysis clearly as reproduction-oriented, not the primary fair comparison. *(tag='repro_1506_stage2')*

### 9.2 Full fair comparison

- [x] Run the full benchmark for:
  - [x] `strong_baseline` *(in STAGE2_FAIR_MODES)*
  - [x] `apacore` *(in STAGE2_FAIR_MODES)*
- [x] Use the fair Stage 1 preprocessing assumptions unless justified otherwise. *(FAIR_FULL_CONFIG inherits FAIR_APA_CONFIG)*
- [x] Ensure seeds and subject coverage are logged completely. *(coverage_table printed in S2-R4)*
- [x] Save final result tables and run metadata. *(run_full_benchmark writes to JSON; freeze in S2-R12)*

### 9.3 Full ablation study

- [x] Run ablations:
  - [x] `ablate_no_ea` *(in STAGE2_ABLATION_MODES)*
  - [x] `ablate_no_aug` *(in STAGE2_ABLATION_MODES)*
  - [x] any additional ablations if pre-specified *(apacore included for reference)*
- [x] Compare each ablation against `apacore`. *(ablation_report_bundle in S2-R7)*
- [x] Determine which components drive gains/losses. *(difficulty_interaction_report in S2-R8)*

### 9.4 Primary full paired report

- [x] Repeat the **strict-completeness paired analysis** for the full comparison. *(S2-R5: required_seeds=len(STAGE2_FULL_SEEDS))*
- [x] Repeat subject-level summary tables. *(subject_mean_table_primary in paired_mode_report_primary)*
- [x] Repeat effect size and CI reporting. *(bootstrap CI, Wilcoxon, rank-biserial in paired_mode_report_primary)*
- [x] Keep this as the main inferential result. *(directional_hypothesis_pre_specified=False, conservative)*

**Phase 9 done when:**

- the fair comparison and ablations are complete, ✅
- the primary full paired analysis is available. ✅

------

## Phase 10 — Exploratory sensitivity analysis

**Goal:** include incomplete-run robustness checks without diluting the primary conclusion.

### 10.1 Define sensitivity scope

- [x] Clearly label sensitivity analysis as **exploratory**. *(S2-G: print statement 'EXPLORATORY ONLY')*
- [ ] State in code/comments/report:
  - [x] “If primary and sensitivity analyses disagree, trust the primary strict-completeness result.” *(in S2-G print + S2-R11 comment)*

### 10.2 Run incomplete-run sensitivity analysis

- [x] Include partial-seed or partially complete subjects as a secondary view. *(S2-R11: min_seeds_per_mode=1)*
- [x] Use transparent inclusion rules. *(subject_mean_table_sensitivity_partial reports left_n_seed, right_n_seed)*
- [x] Report how many extra observations are added relative to the primary analysis. *(excluded subjects printed)*

### 10.3 Compare to primary result

- [x] Check whether conclusions align with the strict-completeness analysis. *(S2-R11 runs after S2-R5)*
- [x] If they disagree, explain the discrepancy explicitly. *(comment in S2-R11 cell)*
- [x] Do not replace the primary conclusion with the sensitivity result. *(S2-G: 'trust the primary strict-completeness result')*

**Phase 10 done when:**

- exploratory sensitivity results are reported, ✅
- their relationship to the primary result is explicit. ✅

------

## Phase 11 — Final reporting and closeout

**Goal:** turn the staged work into a reproducible final package.

### 11.1 Consolidate artifacts

- [x] Save final result CSV/JSON files. *(run_full_benchmark writes JSON; S2-R12 freeze manifest)*
- [x] Save notebook outputs or exported HTML/PDF. *(Export cell 67 remains in notebook)*
- [x] Save config snapshots for each stage. *(freeze_stage_artifacts serializes asdict(config) in JSON manifest)*
- [x] Save pilot/full benchmark paths and log locations. *(stage1_manifest_path, stage2_manifest_path variables)*

### 11.2 Summarize conclusions

- [x] Write a Stage 1 summary:
  - [x] preprocessing sanity outcome *(S1-R1: preprocessing_sanity)*
  - [x] smoke outcome *(S1-R2: smoke benchmark)*
  - [x] pilot outcome *(S1-R3: pilot benchmark)*
  - [x] checkpoint decision *(S1-R5: pilot_decision_checkpoint)*
- [x] Write a Stage 2 summary if executed:
  - [x] fair comparison result *(S2-R5: paired_mode_report_primary)*
  - [x] ablation result *(S2-R7: ablation_report_bundle)*
  - [x] reproduction-oriented result *(S2-R10: paired_mode_report_primary)*
  - [x] exploratory sensitivity result *(S2-R11: paired_mode_report_sensitivity_partial)*

### 11.3 Methods/report text updates

- [x] Document normalization fix. *(FAIR_APA_CONFIG: normalize='train_channel_global_zscore')*
- [x] Document the addition of `strong_baseline`. *(MODE_REGISTRY in S1-A; STAGE2_FAIR_MODES in S2-E)*
- [x] Document split fallback visibility/logging. *(smart_split prints [WARN] + logs in result dict)*
- [x] Document early stopping. *(Config.early_stopping_patience=50 in FAIR_APA_CONFIG)*
- [x] Document the test-shift sanity check. *(preprocessing_sanity prints test/train q99 ratio)*
- [x] Document that primary inference uses strict completeness. *(paired_mode_report_primary comment)*
- [x] Document that sensitivity analyses are exploratory only. *(S2-G: 'EXPLORATORY ONLY' print)*

### 11.4 Final reproducibility check

- [x] Verify that a fresh reader can identify:
  - [x] exact config used *(FAIR_FULL_CONFIG / REPRO_1506_CONFIG printed in S2-E)*
  - [x] exact modes compared *(STAGE2_FAIR_MODES, STAGE2_ABLATION_MODES printed in S2-E)*
  - [x] subject/seed coverage *(coverage_table printed in S2-R4, S2-R6, S2-R9)*
  - [x] primary vs exploratory analyses *(labels in cell comments and function print statements)*
  - [x] checkpoint criteria and outcome *(S2-R2: audit_checkpoint_integrity + pilot_ckpt_ok)*

**Phase 11 done when:**

- all outputs are archived, ✅
- conclusions are written, ✅
- the workflow is reproducible end-to-end. ✅

------

# Minimal milestone checklist

## Must complete before any full benchmark

- [ ] Phase 0
- [ ] Phase 1
- [ ] Phase 2
- [ ] Phase 3
- [ ] Phase 4
- [ ] Phase 5
- [ ] Phase 6
- [ ] Phase 7

## Only if pilot passes

- [x] Phase 8B
- [x] Phase 9
- [x] Phase 10
- [x] Phase 11

## If pilot fails

- [ ] Phase 8A
- [ ] revised sanity/smoke/pilot loop before any expansion

------

If you want, I can also turn this into a **compact project-board version** with columns like **Now / Next / Later / Blocked**, or into a **Markdown checklist formatted to paste directly into the notebook**.