# cv_edge_qc

Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing.

This repository now implements:
- multi-dataset defect detection (`neu`, `dagm`, `ksdd`)
- CPU-only training/inference and edge-profile emulation via thread limits
- classification and YOLO pruning
- ONNX export/benchmarking and real-time simulation
- Streamlit video simulation for interactive demos

## Platform
- Windows 10/11 (PowerShell)
- CPU-only runtime

## Run Full Pipeline
```powershell
.\run.ps1
```

Note: TFLite export dependencies (`onnx2tf`, `tf_keras`, `onnx-graphsurgeon`, `ai-edge-litert`) are installed by `scripts\01_setup_env.ps1`. If export fails with missing-module errors, rerun setup first.

Or step by step:
```powershell
.\scripts\01_setup_env.ps1 - Setup the Requirements
.\scripts\02_get_data.ps1 - Download the specific datasets
.\scripts\03_preprocess.ps1 - Preprocess and aguments the data
.\scripts\04_train.ps1 - trains all the models
.\scripts\05_export.ps1 - exports the results
.\scripts\06_benchmark.ps1 - used to benchmark 
.\scripts\07_make_reports.ps1 - plots the graphs and tables
.\scripts\08_streamlit.ps1 - live stream testing
```

## The Streamlit app supports:
- classification stream simulation
- detection stream simulation
- dataset stream source
- uploaded video source

## Detection Labels by Dataset
- `ksdd`: mask-to-bbox conversion
- `dagm`: `_label` mask-to-bbox conversion
- `neu`: weak-label full-frame bbox using class labels

## Optimization
- Classification pruning
- YOLO pruning: `best_pruned.pt` generated and benchmarked
- ONNX export: baseline + pruned variants
- INT8 dynamic quantization for classification ONNX

## Main Outputs
- Metrics JSON: `outputs/metrics/`
- Exported models: `outputs/models_onnx/`, `outputs/models_tflite/`
- Figures: `docs/figures/`
- Tables: `docs/tables/`

## Demo Data
- Curated demo subsets are available in `data/demo_subset/` for quick validation.
- Full datasets can still be downloaded and prepared with `.\scripts\02_get_data.ps1`.

## Kaggle API Token
Place `kaggle.json` in repo root or:
```text
%USERPROFILE%\.kaggle\kaggle.json
```
Use `scripts\kaggle.json.template` as a template.
