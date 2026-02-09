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

## 7) Training robustness improvements
- Classification training now uses configurable augmentation (flip, rotation, color jitter):
  - `configs/experiment_default.yaml`
  - `src/train/train_classification.py`
- Best-epoch metric tracking and class-index metadata are persisted for reliable downstream evaluation.

## 8) Pruning quality control
- Pruning now includes:
  - per-layer pruning caps
  - optional short finetuning pass
  - acceptance gating using maximum allowed validation accuracy drop
- Code:
  - `configs/experiment_default.yaml`
  - `src/models/prune_models.py`

## 9) Runtime export and benchmark parity
- Classification TFLite export switched from proxy Keras models to trained ONNX -> TFLite conversion when `onnx2tf` is available.
- Detection TFLite artifacts are collected into `outputs/models_tflite/` and benchmarked.
- Benchmarking now records runtime-specific parity fields (`reference_pt_accuracy`, `accuracy_gap_vs_pt`) and separates ONNX fp32/int8 results.
- Code:
  - `src/export/export_models.py`
  - `src/bench/run_benchmarks.py`

## 10) Trade-off ranking and recommendations
- Reporting now adds:
  - weighted ranking table across accuracy-latency-energy
  - per-dataset recommended deployment tables
  - correlation analysis table
  - classification parity-check table
- Code:
  - `src/reports/make_reports.py`
- Outputs:
  - `docs/tables/TABLE_Model_Weighted_Ranking.csv`
  - `docs/tables/TABLE_Detection_Recommended_Deployment.csv`
  - `docs/tables/TABLE_Classification_Recommended_Deployment.csv`
  - `docs/tables/TABLE_Correlation_Analysis.csv`
  - `docs/tables/TABLE_Classification_Parity_Check.csv`
