# cv_edge_qc

Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing.

Scope: Classification + YOLO-based real-time defect detection, edge-optimised, reproducible, defense-ready.
Platform: Windows 10/11 only (PowerShell only).

## One-Command Run
```powershell
.\run.ps1
```

Final output:
```
DEFENSE PACK READY: docs\defense_pack
```

## Repository Structure
```
cv_edge_qc/
  README.md
  requirements.txt
  run.ps1
  configs/
  data/
  outputs/
  docs/
  scripts/
  src/
```

## Kaggle API Token
Place `kaggle.json` in the repo root or at:
```
%USERPROFILE%\.kaggle\kaggle.json
```
Use `scripts\kaggle.json.template` as a starting point.

## Notes
- Windows only, PowerShell only.
- CPU-only inference with edge emulation profiles.
- YOLO detection runs on CPU; export to ONNX/TFLite attempted.
- Any dataset without bounding boxes/masks is excluded from detection and logged.
