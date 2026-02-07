# Proposal Implementation Alignment

This project implementation now aligns to the proposal requirements through concrete code changes.

## 1) Real-time video simulation
- Streamlit app for CPU-only real-time simulation:
  - `src/infer/streamlit_app.py`
  - `scripts/08_streamlit.ps1`
- Supports dataset stream and uploaded video for:
  - classification
  - detection

## 2) Training and validation graphs
- Classification history and figures:
  - `src/train/train_classification.py`
  - Outputs:
    - `docs/figures/FIG_Classification_Loss_*`
    - `docs/figures/FIG_Classification_ValAcc_*`
    - `docs/tables/TABLE_Classification_TrainVal_History.csv`
- YOLO training figures copied/generated from run artifacts:
  - `src/train/train_yolo.py`

## 3) Multi-dataset defect detection
- Detection preprocessing now enabled for:
  - `neu` (weak full-frame labels from class labels)
  - `dagm` (mask-to-bbox via `Label/*_label`)
  - `ksdd` (mask-to-bbox)
- Code:
  - `src/data/preprocess.py`
  - `configs/experiment_default.yaml` (`enabled_detection: true` for all datasets)

## 4) YOLO pruning support
- Detection pruning implemented:
  - `src/models/prune_models.py`
  - Produces `best_pruned.pt`
- Export and benchmark support for pruned YOLO variants:
  - `src/export/export_models.py`
  - `src/bench/run_benchmarks.py`

## 5) CPU-only deployment path
- Training/inference/benchmark workflows remain CPU-only:
  - `src/train/train_classification.py`
  - `src/train/train_yolo.py`
  - `src/bench/run_benchmarks.py`
  - `src/infer/realtime_sim.py`

## 6) Repository cleanup
- Removed non-project thesis/papers/result dump artifacts.
- Cleaned and regenerated project figures/tables under:
  - `docs/figures/`
  - `docs/tables/`
