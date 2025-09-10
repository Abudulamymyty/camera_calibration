import sys
from pathlib import Path

# YAML config
try:
    import yaml  # type: ignore
except Exception as e:
    print("error: PyYAML not installed. Please install pyyaml (pip install pyyaml)")
    sys.exit(2)

# Expand import paths to reach integrated pipeline and shared detector
ROOT = Path(__file__).resolve().parent
ORIG_DIR = ROOT / "original_python_code"
PYDET_DIR = ROOT / "python_detec"
for p in (PYDET_DIR, ORIG_DIR, ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from python_detec.calibration_pipeline_integrated import (
    CalibrationConfig as PostCfg,
    IntegratedCalibrationPipeline,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: Path | None = None) -> int:
    cfg_path = cfg_path or (ROOT / "config.yaml")
    if not cfg_path.is_file():
        print(f"error: config not found -> {cfg_path}")
        return 2

    cfg = load_yaml(cfg_path)
    paths = cfg.get("paths", {})
    root_dir = Path(paths.get("output_dir", "./python_detec")).resolve()

    pcfg = cfg.get("postprocess", {})

    # Build config for integrated pipeline
    post_cfg = PostCfg(
        image_path=str(root_dir / "pic" / "rgb_behindexam_pic.png"),
        board_size=tuple(pcfg.get("board_size", [4, 22])),
        square_size=0.05,
        tau=float(pcfg.get("tau", 0.02)),
        radius_list=None,
        target_boards=1,
        gamma=float(pcfg.get("gamma", 1.2)),
        max_iterations=int(pcfg.get("max_iterations", 10)),
        maxAcceptableError=float(pcfg.get("maxAcceptableError", 10.0)),
        max_error=float(pcfg.get("max_error", 20.0)),
        output_dir=str(root_dir / "Light_AI"),
        visualize=False,
    )

    pipeline = IntegratedCalibrationPipeline(post_cfg)

    process_both = bool(pcfg.get("process_both", True))
    result = pipeline.run_complete_pipeline(root_dir, process_both=process_both)

    return 0 if result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
import sys
from pathlib import Path
from typing import Optional

# YAML config
ttry = None
try:
    import yaml  # type: ignore
except Exception as _e:
    ttry = _e

ROOT = Path(__file__).resolve().parent
PYDET_DIR = ROOT / "python_detec"
for p in (PYDET_DIR, ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

try:
    from python_detec.calibration_pipeline_integrated import run_post_processing  # type: ignore
except Exception:
    # Fallback to direct file import
    try:
        import importlib.util
        mod_path = PYDET_DIR / "calibration_pipeline_integrated.py"
        spec = importlib.util.spec_from_file_location("calibration_pipeline_integrated", str(mod_path))
        m = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(m)
        run_post_processing = getattr(m, "run_post_processing")  # type: ignore
    except Exception as e2:
        print(f"error: cannot import integrated pipeline: {e2}")
        sys.exit(2)


def load_yaml(path: Path) -> dict:
    if ttry is not None and 'yaml' not in sys.modules:
        raise RuntimeError(
            "PyYAML not installed. Please install pyyaml or run: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: Optional[Path] = None) -> int:
    cfg_path = cfg_path or (ROOT / "config.yaml")
    if not cfg_path.is_file():
        print(f"error: config not found -> {cfg_path}")
        return 2

    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", "./python_detec")).resolve()
    dst_root = output_dir

    post = cfg.get("postprocess", {})
    dataset = str(post.get("dataset", "rgb")).lower()

    if dataset == "rgb":
        xml_name = post.get("rgb", {}).get("xml_name", "rgb_Light_AI_Result.xml")
        pixelpoints_name = post.get("rgb", {}).get("pixelpoints_name", "rgb_Pixelpoints.txt")
        out_bin_name = post.get("rgb", {}).get("out_bin_name", "RGBAICameracfg.bin")
        use_fisheye = bool(post.get("rgb", {}).get("use_fisheye", True))
    else:
        xml_name = post.get("std", {}).get("xml_name", "Light_AI_Result.xml")
        pixelpoints_name = post.get("std", {}).get("pixelpoints_name", "Pixelpoints.txt")
        out_bin_name = post.get("std", {}).get("out_bin_name", "AICameracfg.bin")
        use_fisheye = bool(post.get("std", {}).get("use_fisheye", False))

    actualpoints_name = post.get("actualpoints_name", "Actualpoints.txt")

    # Ensure required files exist in Light_AI
    la = dst_root / "Light_AI"
    required = [la / xml_name, la / pixelpoints_name, la / actualpoints_name]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("error: missing files:\n" + "\n".join(missing))
        return 1

    rc = run_post_processing(
        dst_root=dst_root,
        xml_name=xml_name,
        pixelpoints_name=pixelpoints_name,
        actualpoints_name=actualpoints_name,
        max_iterations=int(post.get("max_iterations", 10)),
        maxAcceptableError=float(post.get("maxAcceptableError", 10.0)),
        max_error=float(post.get("max_error", 20.0)),
        out_bin_name=out_bin_name,
        use_fisheye=use_fisheye,
        y_offset=float(post.get("y_offset", 0.0)),
    )

    return 0 if rc == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
