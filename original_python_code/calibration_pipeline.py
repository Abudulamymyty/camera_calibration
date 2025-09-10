import os
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET
# matplotlib no longer required (comparison figure removed)

# Local modules
from enhanced_chessboard_module import EnhancedCornerChessboardDetector
from rearrange_corners import rearrange_corners


# =============================
# Config and Pipeline classes
# =============================
@dataclass
class CalibrationConfig:
    image_path: str = "./calib_pic_1.png"
    board_size: Tuple[int, int] = (7, 8)  # rows, cols of inner corners detected by our algorithm
    square_size: float = 0.025  # meters
    tau: float = 0.02  # corner threshold (lower = more corners, higher = fewer corners)
    radius_list: List[int] = None  # detection radii for multi-scale analysis (default: [3, 5, 7])
    target_boards: int = 5
    gamma: float = 0.6  # basic gamma correction to enhance contrast
    output_dir: str = "data"
    print_key_only: bool = True  # print only key results
    min_boards_required: int = 3  # minimum number of boards needed for calibration


class CalibrationPipeline:
    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        # Always use enhanced detector
        radius_list = self.cfg.radius_list if self.cfg.radius_list is not None else [3, 5, 7]
        self.detector = EnhancedCornerChessboardDetector(tau=self.cfg.tau, radius_list=radius_list)

    # -----------------
    # World coordinates
    # -----------------
    @staticmethod
    def _generate_checkerboard_points(board_size_squares: Tuple[int, int], square_size: float) -> np.ndarray:
        """
        Generate planar (z=0) chessboard points for one board in a canonical frame.
        IMPORTANT: board_size_squares are the number of squares (not inner corners),
        matching original.py. Inner corners are (board_size_squares - 1) each dim.
        """
        # inner corners per dimension
        corners_x = int(board_size_squares[0]) - 1
        corners_y = int(board_size_squares[1]) - 1
        obj_p = np.zeros((corners_x * corners_y, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
        obj_p *= float(square_size)
        return obj_p

    # -------------
    # Calibration
    # -------------
    def _calibrate_fisheye(self, image_points_cell: List[np.ndarray], world_points_cell: List[np.ndarray], image_size: Tuple[int, int]):
        """
        Robust fisheye calibration with multiple fallback strategies for ill-conditioned matrices.
        """
        w, h = image_size

        object_points = []
        image_points = []
        for img_pts, world_pts in zip(image_points_cell, world_points_cell):
            img_pts_formatted = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
            world_pts_formatted = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 3)
            image_points.append(img_pts_formatted)
            object_points.append(world_pts_formatted)

        # Strategy 1: Conservative flags without CALIB_CHECK_COND
        conservative_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW)
        
        # Strategy 2: More relaxed flags
        relaxed_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        
        # Strategy 3: Fixed intrinsics approach
        fixed_flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS + 
                      cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                      cv2.fisheye.CALIB_FIX_SKEW)

        # Initial guess with more reasonable focal lengths
        K0 = np.eye(3, dtype=np.float32)
        # Use image dimensions to estimate better initial focal length
        estimated_focal = max(w, h) * 0.8  # More conservative estimate
        K0[0, 0] = estimated_focal
        K0[1, 1] = estimated_focal
        K0[0, 2] = w / 2.0
        K0[1, 2] = h / 2.0
        D0 = np.zeros((4, 1), dtype=np.float32)

        # Try multiple calibration strategies
        strategies = [
            (conservative_flags, "Conservative (no CALIB_CHECK_COND)"),
            (relaxed_flags, "Relaxed flags"),
            (fixed_flags, "Fixed intrinsics guess")
        ]
        
        for flags, strategy_name in strategies:
            try:
                if not self.cfg.print_key_only:
                    print(f"Trying fisheye calibration strategy: {strategy_name}")
                
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    object_points,
                    image_points,
                    (w, h),
                    K0.copy(),
                    D0.copy(),
                    flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5),  # Relaxed criteria
                )
                
                # Validate the results
                if np.any(np.isnan(K)) or np.any(np.isnan(D)) or np.any(np.isinf(K)) or np.any(np.isinf(D)):
                    raise ValueError("Invalid calibration results (NaN or Inf)")
                
                # Check if focal lengths are reasonable
                fx, fy = K[0, 0], K[1, 1]
                if fx <= 0 or fy <= 0 or fx > 10 * max(w, h) or fy > 10 * max(w, h):
                    raise ValueError(f"Unreasonable focal lengths: fx={fx:.2f}, fy={fy:.2f}")
                
                if not self.cfg.print_key_only:
                    print(f"Fisheye calibration successful with {strategy_name}")
                
                return rms, K, D
                
            except (cv2.error, ValueError) as e:
                if not self.cfg.print_key_only:
                    print(f"Strategy '{strategy_name}' failed: {str(e)}")
                continue
        
        # If all fisheye strategies fail, raise the original error with helpful message
        raise RuntimeError(
            "All fisheye calibration strategies failed. This usually indicates:\n"
            "1. Insufficient corner detection quality\n"
            "2. Too few chessboard views\n"
            "3. Poor chessboard geometry (corners too close or colinear)\n"
            f"Detected boards: {len(image_points_cell)}\n"
            "Consider: lowering tau threshold, capturing more images, or improving lighting")

    # -----------------
    # Undistortion only
    # -----------------
    def _undistort_and_save_fisheye(self, img: np.ndarray, K: np.ndarray, D: np.ndarray) -> str:
        # Direct fisheye undistortion using K as Knew
        try:
            undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=K)
        except cv2.error:
            # Minimal fallback to avoid crash: keep original image
            undistorted = img.copy()

        # Save undistorted image
        out_path = os.path.join(self.cfg.output_dir, "undistorted_fisheye_result.png")
        cv2.imwrite(out_path, undistorted)
        return out_path

    # ------------------------------
    # Write intrinsics to result XML
    # ------------------------------
    def _update_intrinsics_xml(self, K: np.ndarray, D: np.ndarray) -> Optional[str]:
        """Update intrinsics_output in data XML as: cx;cy;fx;k1;k2;k3;k4; (pad D to 4)."""
        xml_path = os.path.join(self.cfg.output_dir, "rgb_Light_AI_Result.xml")
        if not os.path.exists(xml_path):
            return None

        # Extract intrinsics
        fx = float(K[0, 0])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # Distortion coefficients: take first 4 and pad with zeros if needed
        d = np.asarray(D, dtype=float).ravel().tolist()
        while len(d) < 4:
            d.append(0.0)
        d = d[:4]

        # Format with 10 decimals and trailing semicolon
        def f10(x: float) -> str:
            return f"{x:.10f}"

        parts = [f10(cx), f10(cy), f10(fx), f10(d[0]), f10(d[1]), f10(d[2]), f10(d[3])]
        intrinsics_text = ";".join(parts) + ";"

        # Update or create node
        tree = ET.parse(xml_path)
        root = tree.getroot()
        node = root.find("intrinsics_output")
        if node is None:
            node = ET.SubElement(root, "intrinsics_output")
        node.text = intrinsics_text
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path

    # ======
    # Runner
    # ======
    def run(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        t0 = time.time()
        # Load image
        img = cv2.imread(self.cfg.image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.cfg.image_path}")

        # Simple gamma correction to improve edge contrast
        if self.cfg.gamma is not None and self.cfg.gamma > 0:
            img_f = np.float32(img) / 255.0
            img_gamma = np.power(img_f, float(self.cfg.gamma))
            img_proc = np.uint8(np.clip(img_gamma * 255.0, 0, 255))
        else:
            img_proc = img
        
        # 1) Corner detection (enhanced only)
        corners = self.detector.find_corners_enhanced(img_proc, tau=self.cfg.tau)
        num_corners = len(corners['p']) if 'p' in corners else 0

        # Save corner detection visualization
        self._save_corner_visualization(img_proc, corners)

        # 2) Chessboard extraction
        chessboards = self.detector.find_chessboards(
            corners,
            board_size=self.cfg.board_size,
            image=None,  # keep internal debug off
            debug_visualization=False,
            target_boards=self.cfg.target_boards,
            energy_threshold=-40,
        )

        # Save chessboard detection visualization
        self._save_chessboard_visualization(img_proc, corners, chessboards)

        # 3) Rearrangement
        image_points_cell: List[np.ndarray] = []
        if chessboards:
            image_points_cell = rearrange_corners(chessboards, corners, img_proc)

        # Save rearranged corners visualization
        if image_points_cell:
            self._save_rearranged_visualization(img_proc, image_points_cell)

        # 4) World points: replicate original.py logic exactly
        world_points_cell: List[np.ndarray] = []
        if image_points_cell:
            # Use first board as standard template
            first_cb = chessboards[0]
            rows, cols = first_cb.shape  # inner corners
            # As in original.py: standard uses (rows+1, cols+1) squares
            standard_board_size_squares = (rows + 1, cols + 1)
            standard_world = self._generate_checkerboard_points(standard_board_size_squares, self.cfg.square_size)

            for i, cb in enumerate(chessboards):
                r, c = cb.shape  # inner corners
                if i == 0:
                    # Board 1 uses its own standard
                    world_points_cell.append(standard_world)
                else:
                    # As in original.py: current_boardSize uses swapped (cols+1, rows+1)
                    current_board_size_squares = (c + 1, r + 1)
                    if current_board_size_squares == standard_board_size_squares:
                        world_points_cell.append(standard_world.copy())
                    else:
                        # If different size, try to take first N points, else generate fresh
                        current_points_needed = (current_board_size_squares[0]-1) * (current_board_size_squares[1]-1)
                        if current_points_needed <= len(standard_world):
                            world_points_cell.append(standard_world[:current_points_needed].copy())
                        else:
                            tmp_world = self._generate_checkerboard_points(current_board_size_squares, self.cfg.square_size)
                            world_points_cell.append(tmp_world)

        # Early exit if insufficient data
        if not image_points_cell or not world_points_cell:
            if self.cfg.print_key_only:
                elapsed = time.time() - t0
                print(f"error: no valid chessboard; corners={num_corners}; time={elapsed:.3f}s")
                print(f"viz: {self.cfg.output_dir}/corner_detection_visualization.png, {self.cfg.output_dir}/chessboard_detection_visualization.png")
            return None, None, None

        # Check if we have sufficient boards for calibration
        if len(image_points_cell) < self.cfg.min_boards_required:
            if self.cfg.print_key_only:
                elapsed = time.time() - t0
                print(f"error: insufficient boards (need ≥{self.cfg.min_boards_required}, got {len(image_points_cell)}); corners={num_corners}; time={elapsed:.3f}s")
                print(f"viz: {self.cfg.output_dir}/corner_detection_visualization.png, {self.cfg.output_dir}/chessboard_detection_visualization.png, {self.cfg.output_dir}/rearranged_corners_visualization.png")
            return None, None, None

        # 5) Calibration
        h, w = img_proc.shape[:2]
        
        try:
            rms, K, D = self._calibrate_fisheye(image_points_cell, world_points_cell, (w, h))
        except cv2.error as e:
            if self.cfg.print_key_only:
                elapsed = time.time() - t0
                print(f"error: fisheye calibration failed: {e}; time={elapsed:.3f}s")
                print(f"viz: {self.cfg.output_dir}/corner_detection_visualization.png, {self.cfg.output_dir}/chessboard_detection_visualization.png, {self.cfg.output_dir}/rearranged_corners_visualization.png")
            return None, None, None

        # 6) Undistort and save
        undistorted_path = self._undistort_and_save_fisheye(img, K, D)

        # 7) Update result XML intrinsics_output
        xml_updated = self._update_intrinsics_xml(K, D)

        # Key prints only
        if self.cfg.print_key_only:
            elapsed = time.time() - t0
            print(f"summary: corners={num_corners}, boards={len(image_points_cell)}, rms={rms:.6f}, time={elapsed:.3f}s")
            print(f"save: undistorted={undistorted_path}")
            if xml_updated:
                print(f"xml: {xml_updated}")

        return K, D, rms

    def _save_corner_visualization(self, img: np.ndarray, corners: Dict[str, np.ndarray]) -> str:
        """Save corner detection visualization"""
        img_display = img.copy()
        
        if 'p' in corners and len(corners['p']) > 0:
            for i, (x, y) in enumerate(corners['p']):
                # Draw corner with different colors based on score if available
                if 'score' in corners and len(corners['score']) > i:
                    score = corners['score'][i, 0] if corners['score'].ndim > 1 else corners['score'][i]
                    # Color intensity based on score (higher score = brighter green)
                    intensity = min(255, max(100, int(score * 1000)))
                    color = (0, intensity, 0)
                else:
                    color = (0, 255, 0)
                
                cv2.circle(img_display, (int(x), int(y)), 3, color, 2)
        
        # Add info text
        info_text = f"Corners: {len(corners['p']) if 'p' in corners else 0} (tau={self.cfg.tau})"
        cv2.putText(img_display, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out_path = os.path.join(self.cfg.output_dir, "corner_detection_visualization.png")
        cv2.imwrite(out_path, img_display)
        return out_path

    def _save_chessboard_visualization(self, img: np.ndarray, corners: Dict[str, np.ndarray], 
                                     chessboards: List[np.ndarray]) -> str:
        """Save chessboard detection visualization"""
        img_display = img.copy()
        
        if chessboards and 'p' in corners:
            colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
            
            for board_idx, chessboard in enumerate(chessboards):
                color = colors[board_idx % len(colors)]
                rows, cols = chessboard.shape
                
                # Draw chessboard grid
                for r in range(rows):
                    for c in range(cols):
                        corner_idx = chessboard[r, c]
                        if corner_idx != -1 and corner_idx < len(corners['p']):
                            x, y = corners['p'][corner_idx]
                            cv2.circle(img_display, (int(x), int(y)), 5, color, 2)
                            
                            # Draw grid connections
                            if c > 0:  # Connect to left neighbor
                                left_idx = chessboard[r, c-1]
                                if left_idx != -1 and left_idx < len(corners['p']):
                                    x1, y1 = corners['p'][left_idx]
                                    cv2.line(img_display, (int(x1), int(y1)), (int(x), int(y)), color, 1)
                            
                            if r > 0:  # Connect to top neighbor
                                top_idx = chessboard[r-1, c]
                                if top_idx != -1 and top_idx < len(corners['p']):
                                    x1, y1 = corners['p'][top_idx]
                                    cv2.line(img_display, (int(x1), int(y1)), (int(x), int(y)), color, 1)
                
                # Add board label
                if rows > 0 and cols > 0:
                    first_corner_idx = chessboard[0, 0]
                    if first_corner_idx != -1 and first_corner_idx < len(corners['p']):
                        x, y = corners['p'][first_corner_idx]
                        cv2.putText(img_display, f"Board {board_idx+1} ({rows}x{cols})", 
                                  (int(x)-20, int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add info text
        info_text = f"Chessboards: {len(chessboards)} detected"
        cv2.putText(img_display, info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out_path = os.path.join(self.cfg.output_dir, "chessboard_detection_visualization.png")
        cv2.imwrite(out_path, img_display)
        return out_path

    def _save_rearranged_visualization(self, img: np.ndarray, image_points_cell: List[np.ndarray]) -> str:
        """Save rearranged corners visualization"""
        img_display = img.copy()
        
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
        
        for board_idx, points in enumerate(image_points_cell):
            color = colors[board_idx % len(colors)]
            
            # Draw points in order
            for i, (x, y) in enumerate(points):
                cv2.circle(img_display, (int(x), int(y)), 4, color, 2)
                
                # Draw connections to show rearrangement order
                if i > 0:
                    prev_x, prev_y = points[i-1]
                    cv2.line(img_display, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 1)
                
                # Add point numbers for first few points
                if i < 20:  # Only first 20 to avoid clutter
                    cv2.putText(img_display, str(i), (int(x)+3, int(y)+3), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add board info
            if len(points) > 0:
                first_x, first_y = points[0]
                cv2.putText(img_display, f"Board {board_idx+1}: {len(points)} pts", 
                          (int(first_x), int(first_y)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add info text
        info_text = f"Rearranged: {len(image_points_cell)} boards ready for calibration"
        cv2.putText(img_display, info_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out_path = os.path.join(self.cfg.output_dir, "rearranged_corners_visualization.png")
        cv2.imwrite(out_path, img_display)
        return out_path


if __name__ == "__main__":
    cfg = CalibrationConfig(
        # image_path="./behindexam_pic.png",
        image_path="./calib_pic_1.png",
        # board_size=(4, 22),
        board_size=(7, 8),
        square_size=0.05,
        tau=0.1,
        # radius_list=[3, 5, 7],  # Multi-scale detection radii (adjustable)
        radius_list=[5, 7, 9],  # Multi-scale detection radii (adjustable)
        # target_boards=1,
        target_boards=5,
        gamma=0.6,
        output_dir="data",
        print_key_only=True,
    )
    pipeline = CalibrationPipeline(cfg)
    try:
        pipeline.run()
    except Exception as e:
        print(f"error: {e}")
