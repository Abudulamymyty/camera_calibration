# Calibration + Post-processing Split

This repo now has two entry scripts and a shared YAML config.

- `calibration_main.py`: runs the intrinsics calibration (five-chessboard flow) and writes `Light_AI/rgb_Light_AI_Result.xml`.
- `postprocess_main.py`: reads the XML and runs extrinsics post-processing, producing `AICameracfg.bin` / `RGBAICameracfg.bin` and comparison plots.

## Config
Edit `config.yaml` to point to your input/output roots and parameters. Defaults follow the original scripts.

- Inputs: `./python_detec/pic/*.png`
- Outputs: `./python_detec/Light_AI/*`

## Quick start
1. Install dependencies (PyYAML, OpenCV, NumPy, SciPy, Matplotlib)
2. Run calibration, then post-processing.

## Notes
- Corner detection uses `python_detec/enhanced_chessboard_module.py` as in both originals.
- The intermediate XML is `Light_AI/rgb_Light_AI_Result.xml` (created if missing).
