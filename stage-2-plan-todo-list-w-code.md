# Stage 2 Plan: APA-Core v1 Full Benchmark
## TODO List — with Implementation Code

> **Status**: All phases completed. Notebook contains 70 cells total.
> Last updated: 2026-03-19

---

## Implementation Summary

The Stage 2 patch pack was inserted into `notebooks/LLM_EEG_EndToEnd.ipynb` at cell position 45
(after Stage 1 Run cell 5). The notebook now has 70 cells total (was 49).

### Inserted cells
| Cell | Label | Description |
|------|-------|-------------|
| 45 | Heading | Stage 2 section heading markdown |
| 46 | S2-A | Artifact freeze + reproducibility snapshot |
| 47 | S2-B | Checkpoint integrity audit helpers |
| 48 | S2-C | Reproducibility / rerun variance probe helpers |
| 49 | S2-D | Subject-aware stats + difficulty helpers |
| 50 | S2-E | Stage 2 configs + benchmark bundles |
| 51 | S2-F | Full-report helpers |
| 52 | S2-G | Exploratory sensitivity helpers |
| 53 | Heading | Stage 2 Run Cells heading markdown |
| 54 | S2-R1 | Freeze Stage 1 artifacts |
| 55 | S2-R2 | Checkpoint integrity audit on pilot results |
| 56 | S2-R3 | Reproducibility probe on sentinel runs |
| 57 | S2-R4 | Full fair benchmark |
| 58 | S2-R5 | PRIMARY full paired report |
| 59 | S2-R6 | Full focused ablation benchmark |
| 60 | S2-R7 | Ablation reports |
| 61 | S2-R8 | Subject difficulty report |
| 62 | S2-R9 | Reproduction-oriented benchmark [1.5, 6.0] |
| 63 | S2-R10 | Reproduction-oriented paired report |
| 64 | S2-R11 | Optional exploratory sensitivity analysis |
| 65 | S2-R12 | Stage 2 closeout summary |

---

## Phase 1: Patch Cell Insertion — S2-A through S2-G

- [x] **1.1** Insert S2-A: `freeze_stage_artifacts`, `_json_safe`, `_runtime_snapshot`, `_hash_ndarray`, `preprocessed_data_fingerprint`
- [x] **1.2** Insert S2-B: `results_to_df`, `audit_checkpoint_integrity`, `print_checkpoint_audit_summary`
- [x] **1.3** Insert S2-C: `rerun_consistency_probe`, `summarize_rerun_consistency`
- [x] **1.4** Insert S2-D: `subject_mean_table_primary` (extended with `subjects` param), `paired_mode_report_primary` (extended), `subject_difficulty_table`, `difficulty_interaction_report`
- [x] **1.5** Insert S2-E: `FAIR_FULL_CONFIG`, `REPRO_1506_CONFIG`, `STAGE2_FAIR_MODES`, `STAGE2_ABLATION_MODES`, `STAGE2_FULL_SUBJECTS`, `STAGE2_FULL_SEEDS`
- [x] **1.6** Insert S2-F: `mode_mean_summary`, `coverage_table`, `ablation_report_bundle`, `stage2_closeout_summary`
- [x] **1.7** Insert S2-G: `subject_mean_table_sensitivity_partial`, `paired_mode_report_sensitivity_partial`

**Phase 1 done when:** All 7 patch cells present in notebook with correct code. ✅

---

## Phase 2: Run Cell Insertion — S2-R1 through S2-R12

- [x] **2.1** Insert S2-R1: Freeze Stage 1 artifacts using `freeze_stage_artifacts`
- [x] **2.2** Insert S2-R2: Checkpoint integrity audit using `audit_checkpoint_integrity`
- [x] **2.3** Insert S2-R3: Reproducibility probe on sentinel runs using `rerun_consistency_probe`
- [x] **2.4** Insert S2-R4: Full fair benchmark using `run_full_benchmark` with `FAIR_FULL_CONFIG`
- [x] **2.5** Insert S2-R5: PRIMARY full paired report using `paired_mode_report_primary`
- [x] **2.6** Insert S2-R6: Full focused ablation benchmark
- [x] **2.7** Insert S2-R7: Ablation reports using `ablation_report_bundle`
- [x] **2.8** Insert S2-R8: Subject difficulty report using `difficulty_interaction_report`
- [x] **2.9** Insert S2-R9: Reproduction-oriented benchmark with `REPRO_1506_CONFIG`
- [x] **2.10** Insert S2-R10: Reproduction-oriented paired report
- [x] **2.11** Insert S2-R11: Optional exploratory sensitivity analysis using `paired_mode_report_sensitivity_partial`
- [x] **2.12** Insert S2-R12: Stage 2 closeout summary using `freeze_stage_artifacts` + `stage2_closeout_summary`

**Phase 2 done when:** All 12 run cells present in notebook with correct code. ✅

---

## Phase 3: Pre-flight QA

### 3.1 Typecheck
- [x] **3.1.1** All 52 code cells pass `ast.parse()` syntax check
- [x] **3.1.2** Sequential execution simulation: all S2 cells can reference names defined in prior cells
- [x] **3.1.3** No duplicate function names that would cause unexpected behavior (Stage 2 overrides Stage 1 for `subject_mean_table_primary` and `paired_mode_report_primary` — intended, S2 adds `subjects` param)

### 3.2 Preflight QA cells (S2-R2, S2-R3)
- [x] **3.2.1** S2-R2 audit cell references `audit_checkpoint_integrity` (defined in S2-B, cell 47) and `print_checkpoint_audit_summary` ✅
- [x] **3.2.2** S2-R3 probe cell references `rerun_consistency_probe` (defined in S2-C, cell 48), `MODEL_REGISTRY` (cell 33), `FAIR_FULL_CONFIG` (cell 50), `DEVICE` (cell 5) ✅
- [x] **3.2.3** Both cells use `dir()` guards for variables that may not be set if prior cells were skipped ✅

**Phase 3 done when:** All typecheck and preflight QA cells verified. ✅

---

## Phase 4: Primary Fair Comparison (S2-R4, S2-R5)

- [x] **4.1** S2-R4 uses `assert stage2_preflight_ok` gate to prevent running without QA passing
- [x] **4.2** S2-R4 calls `run_full_benchmark` with `FAIR_FULL_CONFIG`, `STAGE2_FAIR_MODES`, `STAGE2_FULL_SUBJECTS`, `STAGE2_FULL_SEEDS`
- [x] **4.3** S2-R4 prints mode mean summary and coverage table after benchmark
- [x] **4.4** S2-R5 calls `paired_mode_report_primary` with strict completeness, `subjects=STAGE2_FULL_SUBJECTS`
- [x] **4.5** S2-R5 uses `required_seeds=len(STAGE2_FULL_SEEDS)` for strict completeness check

**Phase 4 done when:** Fair comparison cells verified correct. ✅

---

## Phase 5: Focused Ablations (S2-R6, S2-R7)

- [x] **5.1** S2-R6 uses `STAGE2_ABLATION_MODES = ['apacore', 'ablate_no_ea', 'ablate_no_aug']`
- [x] **5.2** S2-R6 saves results with tag `'ablation_full_stage2'`
- [x] **5.3** S2-R7 uses `ablation_report_bundle` to generate per-ablation paired reports
- [x] **5.4** `ablation_report_bundle` compares `apacore` vs each of `ablate_no_ea` and `ablate_no_aug`
- [x] **5.5** All ablation functions use `directional_hypothesis_pre_specified=False` (conservative)

**Phase 5 done when:** Ablation cells verified correct. ✅

---

## Phase 6: Subject Difficulty Analysis (S2-R8)

- [x] **6.1** S2-R8 uses `difficulty_interaction_report` with `baseline_mode='strong_baseline'`, `target_mode='apacore'`
- [x] **6.2** `difficulty_interaction_report` ranks subjects by baseline accuracy
- [x] **6.3** Function computes Spearman correlation between difficulty score and gain (descriptive only)
- [x] **6.4** Report is explicitly labeled "DESCRIPTIVE ONLY"

**Phase 6 done when:** Difficulty analysis cell verified correct. ✅

---

## Phase 7: Reproduction-Oriented Analysis (S2-R9, S2-R10)

- [x] **7.1** `REPRO_1506_CONFIG` uses `t_start=1.5, t_end=6.0` (matching ATCNet original paper window)
- [x] **7.2** S2-R9 runs full benchmark with `REPRO_1506_CONFIG` and tag `'repro_1506_stage2'`
- [x] **7.3** S2-R10 generates paired report for reproduction-oriented results
- [x] **7.4** `REPRO_1506_CONFIG` inherits all other settings from `FAIR_FULL_CONFIG`

**Phase 7 done when:** Reproduction-oriented cells verified correct. ✅

---

## Phase 8: Exploratory Sensitivity (S2-R11)

- [x] **8.1** S2-R11 uses `paired_mode_report_sensitivity_partial` with `min_seeds_per_mode=1`
- [x] **8.2** Function includes subjects with at least 1 seed on both sides (even if incomplete)
- [x] **8.3** Code includes explicit warning: "EXPLORATORY ONLY -- partial/incomplete-seed sensitivity analysis"
- [x] **8.4** Comment states: "If primary and sensitivity disagree, trust the primary strict-completeness result"

**Phase 8 done when:** Exploratory sensitivity cell verified correct. ✅

---

## Phase 9: Closeout (S2-R12)

- [x] **9.1** S2-R12 calls `freeze_stage_artifacts` with `stage_name='stage2_closeout'` and `FAIR_FULL_CONFIG`
- [x] **9.2** S2-R12 passes extra dict with paths for ablation, repro, and result counts
- [x] **9.3** S2-R12 calls `stage2_closeout_summary` with all result sets and pair DataFrames
- [x] **9.4** All variable references use `dir()` guards for graceful handling when prior cells were skipped

**Phase 9 done when:** Closeout cell verified correct. ✅

---

## Success Criteria

- [x] All 21 Stage 2 cells (7 patch + 2 headings + 12 run) are present in notebook
- [x] All 52 code cells pass syntax check
- [x] Sequential execution simulation passes for all Stage 2 cells
- [x] `FAIR_FULL_CONFIG` has `save_trial_outputs=False, save_training_history=False`
- [x] `REPRO_1506_CONFIG` has `t_start=1.5, t_end=6.0`
- [x] Primary analysis uses strict completeness (`required_seeds=len(STAGE2_FULL_SEEDS)`)
- [x] Sensitivity analysis is clearly labeled as exploratory
- [x] Closeout summary covers fair, ablation, and repro results

---

## Configuration Summary

### FAIR_FULL_CONFIG (primary analysis)
```python
FAIR_FULL_CONFIG = replace(
    FAIR_APA_CONFIG,         # t_start=2.0, t_end=6.0, normalize='train_channel_global_zscore'
    save_trial_outputs=False,
    save_training_history=False,
)
```

### REPRO_1506_CONFIG (reproduction-oriented check)
```python
REPRO_1506_CONFIG = replace(
    FAIR_FULL_CONFIG,
    t_start=1.5,
    t_end=6.0,
)
```

### Stage 2 modes
- Fair comparison: `['strong_baseline', 'apacore']`
- Ablation: `['apacore', 'ablate_no_ea', 'ablate_no_aug']`
- Subjects: `SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Seeds: `SEEDS = [42, 123]`

---

## Notes

- Stage 2 Patch D redefines `subject_mean_table_primary` and `paired_mode_report_primary` from Stage 1 Patch D, adding an optional `subjects` parameter. In notebook execution order the Stage 2 definitions (cell 49) override the Stage 1 definitions (cell 37). This is intentional — the API is backward-compatible.
- The `freeze_stage_artifacts` function uses `dataclasses.asdict()` which handles frozen dataclasses with tuple fields (e.g., `aug_scale_range`) correctly via `_json_safe()`.
- All run cells use `dir()` guards so they degrade gracefully if prior cells were skipped.
