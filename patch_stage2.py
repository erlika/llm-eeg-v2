import json
import uuid

def make_cell(cell_type, source):
    if cell_type == 'code':
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": [source]
        }
    else:
        return {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [source]
        }

# ============================================================
# STAGE 2 CELLS
# ============================================================

S2_HEADING = """## 9.6 Stage 2 Patch Pack — APA-Core v1 Full Benchmark <a name="sec9_6"></a>

These cells implement the Stage 2 full benchmark: artifact freeze, checkpoint audit,
reproducibility probes, primary fair comparison, focused ablations, subject difficulty
analysis, reproduction-oriented check, and closeout summary.

**Run order:** S2-A → S2-G (patch cells), then S2-R1 → S2-R12 (run cells).
"""

S2_A = """\
# ==============================================================================
# STAGE 2 PATCH A: Artifact freeze + reproducibility snapshot
# ==============================================================================

import os
import sys
import io
import json as _json_mod
import time
import math
import hashlib
import platform
import contextlib
from pathlib import Path
from dataclasses import asdict, replace
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

try:
    import scipy as _scipy
except Exception:
    _scipy = None

try:
    import sklearn as _sklearn
except Exception:
    _sklearn = None


def _json_safe(obj: Any):
    \"\"\"Convert common notebook/runtime objects into JSON-serializable forms.\"\"\"
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _runtime_snapshot() -> dict:
    \"\"\"Capture a lightweight reproducibility snapshot of the current runtime.\"\"\"
    snap = {
        'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'torch_version': torch.__version__,
        'scipy_version': None if _scipy is None else _scipy.__version__,
        'sklearn_version': None if _sklearn is None else _sklearn.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        'cuda_device_names': (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        ),
        'cudnn_enabled': bool(torch.backends.cudnn.enabled),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'torch_deterministic_algorithms_enabled': (
            bool(torch.are_deterministic_algorithms_enabled())
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
    }
    return snap


def _hash_ndarray(arr: np.ndarray, chunk_bytes: int = 1 << 20) -> str:
    \"\"\"SHA256 digest of an ndarray, streamed in chunks to avoid large temp copies.\"\"\"
    arr_c = np.ascontiguousarray(arr)
    raw = arr_c.view(np.uint8).ravel()
    h = hashlib.sha256()
    for start in range(0, raw.size, chunk_bytes):
        h.update(raw[start:start + chunk_bytes].tobytes())
    return h.hexdigest()[:16]


def _label_count_dict(y: np.ndarray) -> dict:
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _quiet_prepare_data(subject_id: int, mode_name: str, config: Any):
    \"\"\"Run _prepare_data without cluttering notebook output.\"\"\"
    mode_flags = MODE_REGISTRY[mode_name]
    with contextlib.redirect_stdout(io.StringIO()):
        return _prepare_data(subject_id, mode_flags, config)


def preprocessed_data_fingerprint(
    subject_id: int,
    mode_name: str,
    config: Any,
) -> dict:
    \"\"\"
    Fingerprint preprocessed train/val/test arrays + key metadata.
    Useful for checking whether data prep is stable across repeated runs.
    \"\"\"
    X_tr, y_tr, X_va, y_va, X_te, y_te, split_info = _quiet_prepare_data(
        subject_id, mode_name, config
    )

    payload = {
        'subject': int(subject_id),
        'mode': str(mode_name),
        'seed': int(config.seed),
        'normalize': str(config.normalize),
        'requested_val_strategy': split_info['requested_val_strategy'],
        'actual_val_strategy': split_info['actual_val_strategy'],
        'split_fallback': bool(split_info['split_fallback']),
        'val_run_id': split_info['val_run_id'],
        'shape_train': tuple(map(int, X_tr.shape)),
        'shape_val': tuple(map(int, X_va.shape)),
        'shape_test': tuple(map(int, X_te.shape)),
        'hash_X_train': _hash_ndarray(X_tr),
        'hash_y_train': _hash_ndarray(y_tr.astype(np.int64)),
        'hash_X_val': _hash_ndarray(X_va),
        'hash_y_val': _hash_ndarray(y_va.astype(np.int64)),
        'hash_X_test': _hash_ndarray(X_te),
        'hash_y_test': _hash_ndarray(y_te.astype(np.int64)),
        'train_label_counts': _label_count_dict(y_tr),
        'val_label_counts': _label_count_dict(y_va),
        'test_label_counts': _label_count_dict(y_te),
        'train_abs_q99': float(np.percentile(np.abs(X_tr), 99)),
        'test_abs_q99': float(np.percentile(np.abs(X_te), 99)),
        'test_train_q99_ratio': float(
            np.percentile(np.abs(X_te), 99) / (np.percentile(np.abs(X_tr), 99) + 1e-8)
        ),
    }

    combined = hashlib.sha256(
        _json_mod.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    payload['combined_hash'] = combined
    return payload


def freeze_stage_artifacts(
    stage_name: str,
    config: Any,
    results: Optional[list] = None,
    failures: Optional[list] = None,
    benchmark_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    \"\"\"Save a compact manifest with config, runtime info, and benchmark coverage.\"\"\"
    out_dir = Path(config.output_dir) / "stage_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [] if results is None else list(results)
    fail_rows = [] if failures is None else list(failures)

    coverage = []
    for r in result_rows:
        coverage.append({
            'subject': int(r['subject']),
            'seed': int(r['seed']),
            'mode': str(r['mode']),
            'status': str(r.get('status', 'unknown')),
            'accuracy': float(r['accuracy']) if r.get('accuracy') is not None else None,
            'kappa': float(r['kappa']) if r.get('kappa') is not None else None,
        })

    manifest = {
        'stage_name': stage_name,
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'config': _json_safe(asdict(config)),
        'runtime': _json_safe(_runtime_snapshot()),
        'benchmark_path': benchmark_path,
        'n_results': len(result_rows),
        'n_failures': len(fail_rows),
        'coverage': coverage,
        'extra': _json_safe(extra or {}),
    }

    file_name = f"{stage_name}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path = out_dir / file_name
    with open(path, "w", encoding="utf-8") as f:
        _json_mod.dump(_json_safe(manifest), f, indent=2)

    print(f"Saved manifest: {path}")
    return str(path)


print("Stage 2 patch A loaded.")
try:
    print(_json_mod.dumps(_json_safe(_runtime_snapshot()), indent=2)[:800])
except Exception as _e:
    print(f"[WARN] runtime snapshot partial: {_e}")
"""

S2_B = """\
# ==============================================================================
# STAGE 2 PATCH B: Checkpoint integrity audit helpers
# ==============================================================================

def results_to_df(all_results: list) -> pd.DataFrame:
    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in all_results])


def audit_checkpoint_integrity(
    all_results: list,
    metric_tol: float = 1e-12,
    epoch_tol: int = 1,
) -> pd.DataFrame:
    \"\"\"
    Audit result rows for consistency between:
      - best_epoch / best_val_metric
      - checkpoint_epochs / checkpoint_metrics
    \"\"\"
    rows = []
    df = results_to_df(all_results)
    if df.empty:
        return pd.DataFrame()

    for _, row in df.iterrows():
        if row.get('status') != 'ok':
            continue

        ck_epochs = list(row.get('checkpoint_epochs') or [])
        ck_metrics = list(row.get('checkpoint_metrics') or [])
        best_epoch = row.get('best_epoch')
        best_metric = row.get('best_val_metric')

        classification = 'unknown'
        severity = 'review'
        notes = []

        if best_metric is None or (isinstance(best_metric, float) and np.isnan(best_metric)):
            classification = 'missing_best_metric'
            severity = 'high'
            notes.append('best_val_metric missing')
        elif len(ck_epochs) == 0 or len(ck_metrics) == 0:
            classification = 'missing_checkpoint_entries'
            severity = 'high'
            notes.append('checkpoint list empty')
        else:
            best_metric_f = float(best_metric)
            max_ck = float(np.max(ck_metrics))
            metric_match_idx = [
                i for i, m in enumerate(ck_metrics)
                if abs(float(m) - best_metric_f) <= metric_tol
            ]
            epoch_exact = (
                best_epoch is not None and int(best_epoch) in [int(e) for e in ck_epochs]
            )
            epoch_near = (
                best_epoch is not None
                and any(abs(int(best_epoch) - int(e)) <= epoch_tol for e in ck_epochs)
            )

            if max_ck < best_metric_f - metric_tol:
                classification = 'best_metric_missing_from_checkpoints'
                severity = 'high'
                notes.append(
                    f'max(checkpoint_metrics)={max_ck:.6f} < best_val_metric={best_metric_f:.6f}'
                )
            elif metric_match_idx and epoch_exact:
                classification = 'exact_match'
                severity = 'ok'
            elif metric_match_idx and epoch_near:
                classification = 'likely_epoch_index_convention_or_tie'
                severity = 'low'
                notes.append('matching metric exists; epoch differs by <= epoch_tol')
            elif metric_match_idx:
                classification = 'tied_metric_saved_at_different_epoch'
                severity = 'low'
                notes.append('matching metric exists at different checkpoint epoch')
            elif epoch_near:
                classification = 'best_epoch_near_checkpoint_without_metric_match'
                severity = 'medium'
                notes.append('epoch nearly matches but metric does not')
            else:
                classification = 'checkpoint_review_needed'
                severity = 'medium'
                notes.append('no clear exact/near match')

            best_metric = best_metric_f  # use float form below

        rows.append({
            'model': row.get('model', ''),
            'mode': row.get('mode', ''),
            'subject': int(row.get('subject', 0)),
            'seed': int(row.get('seed', 0)),
            'best_epoch': None if best_epoch is None else int(best_epoch),
            'best_val_metric': None if best_metric is None else float(best_metric),
            'checkpoint_epochs': ck_epochs,
            'checkpoint_metrics': [float(x) for x in ck_metrics],
            'classification': classification,
            'severity': severity,
            'notes': "; ".join(notes),
        })

    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        sev_order = {'high': 0, 'medium': 1, 'low': 2, 'ok': 3}
        audit_df = audit_df.sort_values(
            by=['severity', 'subject', 'mode', 'seed'],
            key=lambda s: s.map(sev_order).fillna(9) if s.name == 'severity' else s
        ).reset_index(drop=True)
    return audit_df


def print_checkpoint_audit_summary(audit_df: pd.DataFrame) -> bool:
    \"\"\"
    Print a compact checkpoint audit summary.
    Returns True if no high-severity issues were found.
    \"\"\"
    if audit_df.empty:
        print("No checkpoint audit rows available.")
        return False

    print("=" * 70)
    print("Checkpoint integrity audit")
    print("=" * 70)
    print("By classification:")
    print(audit_df['classification'].value_counts(dropna=False).to_string())
    print("-" * 70)
    print("By severity:")
    print(audit_df['severity'].value_counts(dropna=False).to_string())

    high_df = audit_df[audit_df['severity'] == 'high']
    med_df = audit_df[audit_df['severity'] == 'medium']

    if len(high_df) > 0:
        print("-" * 70)
        print("HIGH severity rows:")
        _disp_cols = ['subject', 'mode', 'seed', 'best_epoch', 'best_val_metric',
                      'checkpoint_epochs', 'checkpoint_metrics', 'classification', 'notes']
        print(high_df[[c for c in _disp_cols if c in high_df.columns]].to_string())

    if len(med_df) > 0:
        print("-" * 70)
        print("MEDIUM severity rows:")
        _disp_cols = ['subject', 'mode', 'seed', 'best_epoch', 'best_val_metric',
                      'checkpoint_epochs', 'checkpoint_metrics', 'classification', 'notes']
        print(med_df[[c for c in _disp_cols if c in med_df.columns]].to_string())

    ok_to_proceed = (len(high_df) == 0)
    print("-" * 70)
    print(f"Checkpoint audit ok_to_proceed = {ok_to_proceed}")
    return ok_to_proceed


print("Stage 2 patch B loaded.")
"""

S2_C = """\
# ==============================================================================
# STAGE 2 PATCH C: Reproducibility / rerun variance probe helpers
# ==============================================================================

def rerun_consistency_probe(
    spec: Any,
    subject_id: int,
    mode_name: str,
    seed: int,
    config: Any,
    device: Any,
    n_repeats: int = 2,
) -> pd.DataFrame:
    \"\"\"
    Repeat the exact same subject/mode/seed run multiple times and compare:
      - accuracy / kappa
      - best_epoch / best_val_metric
      - split metadata
      - preprocessed data fingerprint
    \"\"\"
    rows = []

    for rep in range(1, n_repeats + 1):
        run_config = replace(config, seed=int(seed), save_trial_outputs=False, save_training_history=False)

        fp = preprocessed_data_fingerprint(subject_id, mode_name, run_config)
        result, _, history = run_single(spec, subject_id, mode_name, run_config, device)

        rows.append({
            'repeat': rep,
            'subject': int(subject_id),
            'mode': str(mode_name),
            'seed': int(seed),
            'accuracy': float(result['accuracy']),
            'kappa': float(result['kappa']),
            'best_epoch': None if result.get('best_epoch') is None else int(result['best_epoch']),
            'best_val_metric': None if result.get('best_val_metric') is None else float(result['best_val_metric']),
            'requested_val_strategy': result.get('requested_val_strategy'),
            'actual_val_strategy': result.get('actual_val_strategy'),
            'split_fallback': bool(result.get('split_fallback', False)),
            'val_run_id': result.get('val_run_id'),
            'n_train': int(result.get('n_train', 0)),
            'n_val': int(result.get('n_val', 0)),
            'n_test': int(result.get('n_test', 0)),
            'train_time_sec': float(result.get('train_time_sec', 0.0)),
            'fingerprint': fp['combined_hash'],
            'train_hash': fp['hash_X_train'],
            'val_hash': fp['hash_X_val'],
            'test_hash': fp['hash_X_test'],
            'test_train_q99_ratio': float(fp['test_train_q99_ratio']),
        })

    return pd.DataFrame(rows)


def summarize_rerun_consistency(
    probe_df: pd.DataFrame,
    accuracy_tol_pp: float = 0.5,
) -> dict:
    \"\"\"Summarize run-to-run variability and flag concerning instability.\"\"\"
    if probe_df.empty:
        print("Probe dataframe is empty.")
        return {'ok_to_proceed': False}

    acc_pp = probe_df['accuracy'].to_numpy(dtype=float) * 100.0
    spread_pp = float(acc_pp.max() - acc_pp.min())

    unique_fp = sorted(set(probe_df['fingerprint'].tolist()))
    unique_split_sig = sorted(set(
        probe_df.apply(
            lambda r: (
                f"{r['requested_val_strategy']}|{r['actual_val_strategy']}"
                f"|{int(r['split_fallback'])}|{r['val_run_id']}"
            ),
            axis=1
        ).tolist()
    ))

    print("=" * 70)
    print("Rerun consistency summary")
    print("=" * 70)
    print(probe_df.to_string())

    print("-" * 70)
    print(f"Accuracy mean +/- std (pp): {acc_pp.mean():.2f} +/- {acc_pp.std(ddof=0):.2f}")
    print(f"Accuracy min/max (pp):      {acc_pp.min():.2f} / {acc_pp.max():.2f}")
    print(f"Accuracy spread (pp):       {spread_pp:.2f}")
    print(f"Unique data fingerprints:   {len(unique_fp)} -> {unique_fp}")
    print(f"Unique split signatures:    {len(unique_split_sig)} -> {unique_split_sig}")

    warnings_list = []
    if len(unique_fp) != 1:
        warnings_list.append("Data fingerprint changed across reruns.")
    if len(unique_split_sig) != 1:
        warnings_list.append("Split metadata changed across reruns.")
    if spread_pp > accuracy_tol_pp:
        warnings_list.append(
            f"Accuracy spread exceeded tolerance: {spread_pp:.2f} pp > {accuracy_tol_pp:.2f} pp."
        )

    if warnings_list:
        print("-" * 70)
        print("[WARN] Probe warnings:")
        for w in warnings_list:
            print(f"  - {w}")

    ok = (len(unique_fp) == 1) and (len(unique_split_sig) == 1)
    print("-" * 70)
    print(f"ok_to_proceed = {ok}")
    return {
        'ok_to_proceed': ok,
        'accuracy_spread_pp': spread_pp,
        'n_unique_fingerprint': len(unique_fp),
        'n_unique_split_signature': len(unique_split_sig),
        'warnings': warnings_list,
    }


print("Stage 2 patch C loaded.")
"""

S2_D = """\
# ==============================================================================
# STAGE 2 PATCH D: Subject-aware stats + difficulty helpers
# ==============================================================================

from scipy.stats import spearmanr as _spearmanr
from scipy.stats import wilcoxon as _wilcoxon_scipy
from scipy.stats import binomtest as _binomtest


def subject_mean_table_primary(
    all_results: list,
    left_mode: str,
    right_mode: str,
    required_seeds: int,
    subjects: Optional[list] = None,
):
    rows = []
    excluded = []

    if subjects is None:
        subjects = sorted({int(r['subject']) for r in all_results if r.get('status') == 'ok'})

    for subj in subjects:
        left = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == left_mode and int(r['subject']) == int(subj)
        ]
        right = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == right_mode and int(r['subject']) == int(subj)
        ]

        if len(left) == required_seeds and len(right) == required_seeds:
            rows.append({
                'subject': int(subj),
                'left_mean': float(np.mean(left)),
                'right_mean': float(np.mean(right)),
                'delta': float(np.mean(right) - np.mean(left)),
                'left_std_seed': float(np.std(left)),
                'right_std_seed': float(np.std(right)),
                'left_n_seed': int(len(left)),
                'right_n_seed': int(len(right)),
            })
        else:
            excluded.append({
                'subject': int(subj),
                'left_n_seed': int(len(left)),
                'right_n_seed': int(len(right)),
            })

    return pd.DataFrame(rows), excluded


def paired_mode_report_primary(
    all_results: list,
    left_mode: str,
    right_mode: str,
    required_seeds: int,
    directional_hypothesis_pre_specified: bool = False,
    subjects: Optional[list] = None,
):
    print("=" * 70)
    print(f"PRIMARY analysis (strict completeness): {right_mode} vs {left_mode}")
    print("=" * 70)

    df_pair, excluded = subject_mean_table_primary(
        all_results=all_results,
        left_mode=left_mode,
        right_mode=right_mode,
        required_seeds=required_seeds,
        subjects=subjects,
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
        sign_p = _binomtest(
            n_positive,
            n_positive + n_negative,
            p=0.5,
            alternative=sign_alt
        ).pvalue
        print(f"Sign test p ({sign_alt}): {sign_p:.4f}")

    if n_nonzero >= 5:
        stat_two, p_two = _wilcoxon_scipy(
            df_pair['right_mean'].to_numpy(),
            df_pair['left_mean'].to_numpy(),
            alternative='two-sided',
            zero_method='wilcox'
        )
        print(f"Wilcoxon statistic: {stat_two:.4f}")
        print(f"Wilcoxon two-sided p: {p_two:.4f}")

        if directional_hypothesis_pre_specified:
            _, p_one = _wilcoxon_scipy(
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


def subject_difficulty_table(
    all_results: list,
    modes: list,
    required_seeds: int,
    subjects: Optional[list] = None,
) -> pd.DataFrame:
    \"\"\"Per-subject mean accuracy table across the requested modes.\"\"\"
    if subjects is None:
        subjects = sorted({int(r['subject']) for r in all_results if r.get('status') == 'ok'})

    rows = []
    for subj in subjects:
        row = {'subject': int(subj)}
        complete = True

        for mode in modes:
            vals = [
                float(r['accuracy']) for r in all_results
                if r.get('status') == 'ok' and r['mode'] == mode and int(r['subject']) == int(subj)
            ]
            row[f'{mode}_mean'] = float(np.mean(vals)) if len(vals) > 0 else np.nan
            row[f'{mode}_std_seed'] = float(np.std(vals)) if len(vals) > 0 else np.nan
            row[f'{mode}_n_seed'] = int(len(vals))
            if len(vals) != required_seeds:
                complete = False

        if complete:
            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values('subject').reset_index(drop=True) if not out.empty else out


def difficulty_interaction_report(
    all_results: list,
    baseline_mode: str = 'strong_baseline',
    target_mode: str = 'apacore',
    required_seeds: int = 2,
    subjects: Optional[list] = None,
):
    \"\"\"Descriptive analysis: rank subjects by baseline difficulty and inspect gains.\"\"\"
    df = subject_difficulty_table(
        all_results=all_results,
        modes=[baseline_mode, target_mode],
        required_seeds=required_seeds,
        subjects=subjects,
    )

    if df.empty:
        print("No complete subjects available for difficulty report.")
        return None

    df['baseline_acc'] = df[f'{baseline_mode}_mean']
    df['target_acc'] = df[f'{target_mode}_mean']
    df['gain_pp'] = (df['target_acc'] - df['baseline_acc']) * 100.0
    df['difficulty_score'] = 1.0 - df['baseline_acc']
    df = df.sort_values('baseline_acc', ascending=True).reset_index(drop=True)

    print("=" * 70)
    print("Subject difficulty report (DESCRIPTIVE ONLY)")
    print("=" * 70)
    print(df[['subject', 'baseline_acc', 'target_acc', 'gain_pp']].round(4).to_string(index=False))

    if len(df) >= 3:
        rho, p = _spearmanr(df['difficulty_score'], df['gain_pp'])
        print("-" * 70)
        print(f"Spearman rho(difficulty, gain_pp): {rho:.4f}")
        print(f"Spearman p-value (descriptive only): {p:.4f}")

    hardest = df.iloc[0]
    print("-" * 70)
    print(
        f"Hardest subject by {baseline_mode}: "
        f"S{int(hardest['subject']):02d} "
        f"(baseline_acc={hardest['baseline_acc'] * 100:.2f}%, gain={hardest['gain_pp']:.2f} pp)"
    )
    print("=" * 70)
    return df


print("Stage 2 patch D loaded.")
"""

S2_E = """\
# ==============================================================================
# STAGE 2 PATCH E: Stage 2 configs + benchmark bundles
# ==============================================================================

FAIR_FULL_CONFIG = replace(
    FAIR_APA_CONFIG,
    save_trial_outputs=False,
    save_training_history=False,
)

REPRO_1506_CONFIG = replace(
    FAIR_FULL_CONFIG,
    t_start=1.5,
    t_end=6.0,
)

STAGE2_FAIR_MODES = ['strong_baseline', 'apacore']
STAGE2_ABLATION_MODES = ['apacore', 'ablate_no_ea', 'ablate_no_aug']
STAGE2_FULL_SUBJECTS = list(SUBJECTS)
STAGE2_FULL_SEEDS = list(SEEDS)

print("Stage 2 patch E loaded.")
print(f"FAIR_FULL_CONFIG: window=[{FAIR_FULL_CONFIG.t_start}, {FAIR_FULL_CONFIG.t_end}] "
      f"normalize={FAIR_FULL_CONFIG.normalize} val_strategy={FAIR_FULL_CONFIG.val_strategy}")
print(f"REPRO_1506_CONFIG: window=[{REPRO_1506_CONFIG.t_start}, {REPRO_1506_CONFIG.t_end}]")
print(f"Fair modes: {STAGE2_FAIR_MODES}")
print(f"Ablation modes: {STAGE2_ABLATION_MODES}")
print(f"Subjects: {STAGE2_FULL_SUBJECTS}")
print(f"Seeds: {STAGE2_FULL_SEEDS}")
"""

S2_F = """\
# ==============================================================================
# STAGE 2 PATCH F: Full-report helpers
# ==============================================================================

def mode_mean_summary(
    all_results: list,
    metric: str = 'accuracy',
) -> pd.DataFrame:
    df = results_to_df([r for r in all_results if r.get('status') == 'ok'])
    if df.empty:
        print("No successful results to summarize.")
        return pd.DataFrame()

    out = (
        df.groupby('mode')[metric]
        .agg(['count', 'mean', 'std', 'min', 'max'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )
    return out


def coverage_table(
    all_results: list,
    modes: Optional[list] = None,
    subjects: Optional[list] = None,
    seeds: Optional[list] = None,
) -> pd.DataFrame:
    df = results_to_df([r for r in all_results if r.get('status') == 'ok'])
    if df.empty:
        return pd.DataFrame()

    if modes is not None:
        df = df[df['mode'].isin(modes)]
    if subjects is not None:
        df = df[df['subject'].isin(subjects)]
    if seeds is not None:
        df = df[df['seed'].isin(seeds)]

    out = (
        df.groupby(['mode', 'subject'])['seed']
        .nunique()
        .reset_index(name='n_seed')
        .pivot(index='subject', columns='mode', values='n_seed')
        .sort_index()
    )
    return out


def ablation_report_bundle(
    all_results: list,
    required_seeds: int,
    subjects: Optional[list] = None,
) -> dict:
    reports = {}
    for compare_mode in ['ablate_no_ea', 'ablate_no_aug']:
        print(f"\\n### Ablation report: apacore vs {compare_mode}")
        reports[compare_mode] = paired_mode_report_primary(
            all_results=all_results,
            left_mode=compare_mode,
            right_mode='apacore',
            required_seeds=required_seeds,
            directional_hypothesis_pre_specified=False,
            subjects=subjects,
        )
    return reports


def stage2_closeout_summary(
    fair_full_results: list,
    fair_pair_df: Optional[pd.DataFrame] = None,
    ablation_results: Optional[list] = None,
    repro_results: Optional[list] = None,
    repro_pair_df: Optional[pd.DataFrame] = None,
) -> None:
    print("=" * 70)
    print("Stage 2 closeout summary")
    print("=" * 70)

    print("\\nFair full mode summary:")
    print(mode_mean_summary(fair_full_results).to_string())

    if fair_pair_df is not None and len(fair_pair_df) > 0:
        diff_pp = fair_pair_df['delta'].to_numpy() * 100.0
        print("-" * 70)
        print(f"Primary fair comparison mean delta (pp): {diff_pp.mean():.2f}")
        print(f"Improved / worsened / tied: "
              f"{int(np.sum(diff_pp > 0))} / {int(np.sum(diff_pp < 0))} / {int(np.sum(diff_pp == 0))}")

    if ablation_results is not None:
        print("\\nAblation mode summary:")
        print(mode_mean_summary(ablation_results).to_string())

    if repro_results is not None:
        print("\\nReproduction-oriented mode summary:")
        print(mode_mean_summary(repro_results).to_string())

    if repro_pair_df is not None and len(repro_pair_df) > 0:
        repro_diff_pp = repro_pair_df['delta'].to_numpy() * 100.0
        print("-" * 70)
        print(f"Reproduction-oriented mean delta (pp): {repro_diff_pp.mean():.2f}")

    print("=" * 70)


print("Stage 2 patch F loaded.")
"""

S2_G = """\
# ==============================================================================
# STAGE 2 PATCH G: Exploratory sensitivity helpers
# ==============================================================================

def subject_mean_table_sensitivity_partial(
    all_results: list,
    left_mode: str,
    right_mode: str,
    min_seeds_per_mode: int = 1,
    subjects: Optional[list] = None,
):
    \"\"\"
    Exploratory only: include subjects with at least min_seeds_per_mode seeds
    on both sides, even if seed counts are incomplete or unequal.
    \"\"\"
    rows = []
    excluded = []

    if subjects is None:
        subjects = sorted({int(r['subject']) for r in all_results if r.get('status') == 'ok'})

    for subj in subjects:
        left = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == left_mode and int(r['subject']) == int(subj)
        ]
        right = [
            float(r['accuracy']) for r in all_results
            if r.get('status') == 'ok' and r['mode'] == right_mode and int(r['subject']) == int(subj)
        ]

        if len(left) >= min_seeds_per_mode and len(right) >= min_seeds_per_mode:
            rows.append({
                'subject': int(subj),
                'left_mean': float(np.mean(left)),
                'right_mean': float(np.mean(right)),
                'delta': float(np.mean(right) - np.mean(left)),
                'left_n_seed': int(len(left)),
                'right_n_seed': int(len(right)),
            })
        else:
            excluded.append({
                'subject': int(subj),
                'left_n_seed': int(len(left)),
                'right_n_seed': int(len(right)),
            })

    return pd.DataFrame(rows), excluded


def paired_mode_report_sensitivity_partial(
    all_results: list,
    left_mode: str,
    right_mode: str,
    min_seeds_per_mode: int = 1,
    subjects: Optional[list] = None,
):
    print("=" * 70)
    print("EXPLORATORY ONLY -- partial/incomplete-seed sensitivity analysis")
    print("If primary and sensitivity disagree, trust the primary strict-completeness result.")
    print("=" * 70)

    df_pair, excluded = subject_mean_table_sensitivity_partial(
        all_results=all_results,
        left_mode=left_mode,
        right_mode=right_mode,
        min_seeds_per_mode=min_seeds_per_mode,
        subjects=subjects,
    )

    if excluded:
        print("Excluded subjects:")
        for e in excluded:
            print(f"  S{e['subject']:02d}: left={e['left_n_seed']} right={e['right_n_seed']}")

    if len(df_pair) == 0:
        print("No subjects available for exploratory sensitivity analysis.")
        return None

    print(df_pair[['subject', 'left_mean', 'right_mean', 'delta', 'left_n_seed', 'right_n_seed']]
          .round(4).to_string(index=False))

    diff = df_pair['delta'].to_numpy()
    print("-" * 70)
    print(f"N subjects included: {len(df_pair)}")
    print(f"Mean delta:   {diff.mean():.4f}")
    print(f"Median delta: {np.median(diff):.4f}")

    ci_low, ci_high = bootstrap_mean_ci(diff)
    print(f"Bootstrap 95% CI for mean delta: [{ci_low:.4f}, {ci_high:.4f}]")
    print("=" * 70)
    return df_pair


print("Stage 2 patch G loaded.")
"""

# ============================================================
# STAGE 2 RUN CELLS
# ============================================================

S2_RUN_HEADING = """### Stage 2 Run Cells

Execute in order after all Stage 2 patch cells (S2-A through S2-G) are loaded.
**S2-R2** and **S2-R3** are preflight QA — do not skip them.
"""

S2_R1 = """\
# ==============================================================================
# STAGE 2 RUN 1: Freeze Stage 1 artifacts
# ==============================================================================

stage1_manifest_path = freeze_stage_artifacts(
    stage_name='stage1_pilot_freeze',
    config=FAIR_APA_CONFIG,
    results=pilot_results if 'pilot_results' in dir() else [],
    failures=pilot_failures if 'pilot_failures' in dir() else [],
    benchmark_path=pilot_path if 'pilot_path' in dir() else None,
    extra={
        'pilot_ok': bool(pilot_ok) if 'pilot_ok' in dir() else None,
        'pilot_subjects': [1, 2, 3],
        'pilot_seeds': [42, 123],
        'pilot_primary_modes': ['strong_baseline', 'apacore'],
    },
)

print(f"stage1_manifest_path = {stage1_manifest_path}")
"""

S2_R2 = """\
# ==============================================================================
# STAGE 2 RUN 2: Checkpoint integrity audit on pilot results
# ==============================================================================

_pilot_results_for_audit = pilot_results if 'pilot_results' in dir() else []
pilot_ckpt_audit_df = audit_checkpoint_integrity(_pilot_results_for_audit)
pilot_ckpt_ok = print_checkpoint_audit_summary(pilot_ckpt_audit_df)

print(f"\\npilot_ckpt_ok = {pilot_ckpt_ok}")
# Stop rule: if pilot_ckpt_ok is False due to high-severity issues, inspect checkpoint logic first.
"""

S2_R3 = """\
# ==============================================================================
# STAGE 2 RUN 3: Reproducibility probe on sentinel runs
# ==============================================================================

if 'atcnet' not in MODEL_REGISTRY:
    print("[WARN] atcnet not in MODEL_REGISTRY — using first available spec")
    _spec_key = list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else None
else:
    _spec_key = 'atcnet'

if _spec_key is not None:
    spec = MODEL_REGISTRY[_spec_key]

    probe_1_df = rerun_consistency_probe(
        spec=spec,
        subject_id=1,
        mode_name='strong_baseline',
        seed=42,
        config=FAIR_FULL_CONFIG,
        device=DEVICE,
        n_repeats=2,
    )
    probe_1_summary = summarize_rerun_consistency(probe_1_df, accuracy_tol_pp=0.5)

    probe_2_df = rerun_consistency_probe(
        spec=spec,
        subject_id=2,
        mode_name='apacore',
        seed=123,
        config=FAIR_FULL_CONFIG,
        device=DEVICE,
        n_repeats=2,
    )
    probe_2_summary = summarize_rerun_consistency(probe_2_df, accuracy_tol_pp=0.5)

    stage2_preflight_ok = (
        bool(pilot_ckpt_ok if 'pilot_ckpt_ok' in dir() else True)
        and bool(probe_1_summary['ok_to_proceed'])
        and bool(probe_2_summary['ok_to_proceed'])
    )
else:
    print("[WARN] No spec available in MODEL_REGISTRY; skipping reproducibility probe.")
    stage2_preflight_ok = bool(pilot_ckpt_ok if 'pilot_ckpt_ok' in dir() else False)

print(f"\\nstage2_preflight_ok = {stage2_preflight_ok}")
# Stop rule: if fingerprints or split metadata differ across repeats, investigate before full run.
"""

S2_R4 = """\
# ==============================================================================
# STAGE 2 RUN 4: Full fair benchmark
# ==============================================================================

assert stage2_preflight_ok, "Preflight QA did not pass. Resolve checkpoint/reproducibility issues first."

fair_full_results, fair_full_failures, fair_full_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=FAIR_FULL_CONFIG,
    modes=STAGE2_FAIR_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
    tag='fair_full_stage2',
)

print(f"\\nFair full saved to: {fair_full_path}")
print(f"Fair full failures: {len(fair_full_failures)}")
print(mode_mean_summary(fair_full_results).to_string())
print(coverage_table(
    fair_full_results,
    modes=STAGE2_FAIR_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
).to_string())
"""

S2_R5 = """\
# ==============================================================================
# STAGE 2 RUN 5: PRIMARY full paired report
# ==============================================================================

fair_full_pair_df = paired_mode_report_primary(
    all_results=fair_full_results,
    left_mode='strong_baseline',
    right_mode='apacore',
    required_seeds=len(STAGE2_FULL_SEEDS),
    directional_hypothesis_pre_specified=False,
    subjects=STAGE2_FULL_SUBJECTS,
)
"""

S2_R6 = """\
# ==============================================================================
# STAGE 2 RUN 6: Full focused ablation benchmark
# ==============================================================================

ablation_results, ablation_failures, ablation_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=FAIR_FULL_CONFIG,
    modes=STAGE2_ABLATION_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
    tag='ablation_full_stage2',
)

print(f"\\nAblation full saved to: {ablation_path}")
print(f"Ablation failures: {len(ablation_failures)}")
print(mode_mean_summary(ablation_results).to_string())
print(coverage_table(
    ablation_results,
    modes=STAGE2_ABLATION_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
).to_string())
"""

S2_R7 = """\
# ==============================================================================
# STAGE 2 RUN 7: Ablation reports
# ==============================================================================

ablation_report_dfs = ablation_report_bundle(
    all_results=ablation_results,
    required_seeds=len(STAGE2_FULL_SEEDS),
    subjects=STAGE2_FULL_SUBJECTS,
)
"""

S2_R8 = """\
# ==============================================================================
# STAGE 2 RUN 8: Subject difficulty report
# ==============================================================================

difficulty_df = difficulty_interaction_report(
    all_results=fair_full_results,
    baseline_mode='strong_baseline',
    target_mode='apacore',
    required_seeds=len(STAGE2_FULL_SEEDS),
    subjects=STAGE2_FULL_SUBJECTS,
)
"""

S2_R9 = """\
# ==============================================================================
# STAGE 2 RUN 9: Reproduction-oriented benchmark [1.5, 6.0]
# ==============================================================================

repro_results, repro_failures, repro_path = run_full_benchmark(
    spec=spec,
    device=DEVICE,
    config=REPRO_1506_CONFIG,
    modes=STAGE2_FAIR_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
    tag='repro_1506_stage2',
)

print(f"\\nReproduction-oriented benchmark saved to: {repro_path}")
print(f"Reproduction-oriented failures: {len(repro_failures)}")
print(mode_mean_summary(repro_results).to_string())
print(coverage_table(
    repro_results,
    modes=STAGE2_FAIR_MODES,
    subjects=STAGE2_FULL_SUBJECTS,
    seeds=STAGE2_FULL_SEEDS,
).to_string())
"""

S2_R10 = """\
# ==============================================================================
# STAGE 2 RUN 10: Reproduction-oriented paired report
# ==============================================================================

repro_pair_df = paired_mode_report_primary(
    all_results=repro_results,
    left_mode='strong_baseline',
    right_mode='apacore',
    required_seeds=len(STAGE2_FULL_SEEDS),
    directional_hypothesis_pre_specified=False,
    subjects=STAGE2_FULL_SUBJECTS,
)
"""

S2_R11 = """\
# ==============================================================================
# STAGE 2 RUN 11: Optional exploratory sensitivity analysis
# ==============================================================================

# Run only after primary reports are complete.
# Use if there are failures or incomplete seed coverage.
fair_full_sensitivity_df = paired_mode_report_sensitivity_partial(
    all_results=fair_full_results,
    left_mode='strong_baseline',
    right_mode='apacore',
    min_seeds_per_mode=1,
    subjects=STAGE2_FULL_SUBJECTS,
)
"""

S2_R12 = """\
# ==============================================================================
# STAGE 2 RUN 12: Stage 2 closeout summary
# ==============================================================================

stage2_manifest_path = freeze_stage_artifacts(
    stage_name='stage2_closeout',
    config=FAIR_FULL_CONFIG,
    results=fair_full_results if 'fair_full_results' in dir() else [],
    failures=fair_full_failures if 'fair_full_failures' in dir() else [],
    benchmark_path=fair_full_path if 'fair_full_path' in dir() else None,
    extra={
        'ablation_path': ablation_path if 'ablation_path' in dir() else None,
        'repro_path': repro_path if 'repro_path' in dir() else None,
        'n_fair_results': len(fair_full_results) if 'fair_full_results' in dir() else None,
        'n_ablation_results': len(ablation_results) if 'ablation_results' in dir() else None,
        'n_repro_results': len(repro_results) if 'repro_results' in dir() else None,
    },
)

stage2_closeout_summary(
    fair_full_results=fair_full_results if 'fair_full_results' in dir() else [],
    fair_pair_df=fair_full_pair_df if 'fair_full_pair_df' in dir() else None,
    ablation_results=ablation_results if 'ablation_results' in dir() else None,
    repro_results=repro_results if 'repro_results' in dir() else None,
    repro_pair_df=repro_pair_df if 'repro_pair_df' in dir() else None,
)

print(f"\\nstage2_manifest_path = {stage2_manifest_path}")
"""

# ============================================================
# PATCH NOTEBOOK
# ============================================================

with open('notebooks/LLM_EEG_EndToEnd.ipynb', 'r') as f:
    nb = json.load(f)

# Insert after cell 44 (last Stage 1 run cell)
insert_pos = 45

new_cells = [
    make_cell('markdown', S2_HEADING),
    make_cell('code', S2_A),
    make_cell('code', S2_B),
    make_cell('code', S2_C),
    make_cell('code', S2_D),
    make_cell('code', S2_E),
    make_cell('code', S2_F),
    make_cell('code', S2_G),
    make_cell('markdown', S2_RUN_HEADING),
    make_cell('code', S2_R1),
    make_cell('code', S2_R2),
    make_cell('code', S2_R3),
    make_cell('code', S2_R4),
    make_cell('code', S2_R5),
    make_cell('code', S2_R6),
    make_cell('code', S2_R7),
    make_cell('code', S2_R8),
    make_cell('code', S2_R9),
    make_cell('code', S2_R10),
    make_cell('code', S2_R11),
    make_cell('code', S2_R12),
]

nb['cells'] = nb['cells'][:insert_pos] + new_cells + nb['cells'][insert_pos:]

with open('notebooks/LLM_EEG_EndToEnd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook patched. Total cells: {len(nb['cells'])}")
print(f"Inserted {len(new_cells)} new cells at position {insert_pos}")
