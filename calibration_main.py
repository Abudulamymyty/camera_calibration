import sys
from pathlib import Path
from typing import Optional

# YAML config
ttry = None
try:
    import yaml  # type: ignore
except Exception as _e:
    ttry = _e

# Expand import path for local modules
ROOT = Path(__file__).resolve().parent
ORIG_DIR = ROOT / "original_python_code"
PYDET_DIR = ROOT / "python_detec"
# Prepend in priority order: python_detec (for shared enhanced_chessboard_module),
# original_python_code (for rearrange_corners etc.), then project root for package-style imports
for p in (PYDET_DIR, ORIG_DIR, ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

# Import original calibration pipeline
try:
    # Prefer package-style import (requires ROOT on sys.path)
    from original_python_code.calibration_pipeline import CalibrationConfig as OrigCalibConfig  # type: ignore
    from original_python_code.calibration_pipeline import CalibrationPipeline  # type: ignore
except Exception:
    # Fallback: direct module import if package import fails
    try:
        import importlib.util
        mod_path = ORIG_DIR / "calibration_pipeline.py"
        spec = importlib.util.spec_from_file_location("calibration_pipeline", str(mod_path))
        cp = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(cp)
        OrigCalibConfig = getattr(cp, "CalibrationConfig")  # type: ignore
        CalibrationPipeline = getattr(cp, "CalibrationPipeline")  # type: ignore
    except Exception as e2:
        print(f"error: cannot import original pipeline: {e2}")
        sys.exit(2)

import xml.etree.ElementTree as ET  # noqa: E402


def load_yaml(path: Path) -> dict:
    if ttry is not None and 'yaml' not in sys.modules:
        raise RuntimeError(
            "PyYAML not installed. Please install pyyaml or run: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_xml_template(xml_path: Path) -> None:
    if xml_path.is_file():
        return
    root = ET.Element("result")
    ET.SubElement(root, "result_code").text = "0"
    ET.SubElement(root, "result_info").text = "Calibration Success"
    ET.SubElement(root, "camera_type").text = "3"
    ET.SubElement(root, "intrinsics_output").text = ""
    ET.SubElement(root, "extrinsics_output").text = "0;0;0;0;0;0;"
    ET.SubElement(root, "reproj_error").text = "0.0;"
    ET.SubElement(root, "angle_error").text = "0;0;0;"
    ET.SubElement(root, "undistort_img").text = ".\\data\\undistort_calib.png"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(xml_path), encoding="utf-8", xml_declaration=True)


def main(cfg_path: Optional[Path] = None) -> int:
    cfg_path = cfg_path or (ROOT / "config.yaml")
    if not cfg_path.is_file():
        print(f"error: config not found -> {cfg_path}")
        return 2

    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {})
    input_dir = Path(paths.get("input_dir", "./python_detec")).resolve()
    output_dir = Path(paths.get("output_dir", "./python_detec")).resolve()

    calib = cfg.get("calibration", {})
    image_path = calib.get("image_path") or ""
    if not image_path:
        cand1 = input_dir / "pic" / "rgb_behindexam_pic.png"
        cand2 = input_dir / "pic" / "behindexam_pic.png"
        image_path = str(cand1 if cand1.is_file() else cand2)
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"error: image not found -> {img_path}")
        return 1

    # Outputs under Light_AI inside output_dir
    out_subdir = calib.get("output_subdir", "Light_AI")
    final_out_dir = output_dir / out_subdir
    final_out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare xml in target folder so original pipeline can update intrinsics_output
    xml_path = final_out_dir / "rgb_Light_AI_Result.xml"
    ensure_xml_template(xml_path)

    # Build config for original pipeline
    board_size = tuple(calib.get("board_size", [7, 8]))
    radius_list = calib.get("radius_list", [5, 7, 9])
    ocfg = OrigCalibConfig(
        image_path=str(img_path),
        board_size=board_size,
        square_size=float(calib.get("square_size", 0.05)),
        tau=float(calib.get("tau", 0.1)),
        radius_list=list(radius_list),
        target_boards=int(calib.get("target_boards", 5)),
        gamma=float(calib.get("gamma", 0.6)),
        output_dir=str(final_out_dir),
        print_key_only=True,
        min_boards_required=int(calib.get("min_boards_required", 3)),
    )

    pipe = CalibrationPipeline(ocfg)
    try:
        K, D, rms = pipe.run()
    except Exception as e:
        print(f"error: {e}")
        return 3

    # If xml wasn't updated for some reason, try writing intrinsics now
    try:
        if xml_path.is_file():
            pipe._update_intrinsics_xml(K, D)  # type: ignore[attr-defined]
    except Exception:
        pass

    print(f"summary: xml saved -> {xml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
