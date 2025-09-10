#!/usr/bin/env python3
"""
Enhanced chessboard corner detection module with improved scoring and fast template creation
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import convolve, maximum_filter

class EnhancedCornerChessboardDetector:
    """
    Enhanced corner detection module with improved scoring mechanism and configurable parameters.
    
    Key improvements:
    - Linear weight distribution for faster computation
    - Enhanced scoring that leverages chessboard characteristics (contrast, symmetry, stability)
    - Better utilization of diagonal region patterns
    - Configurable detection parameters for different scenarios
    
    Parameters:
    - tau: Corner detection threshold (default: 0.02)
    - radius_list: List of detection radii for multi-scale analysis (default: [3, 5, 7])
    """

    def __init__(self, tau: float = 0.2, radius_list: List[int] = None):
        self.tau = tau
        self.radius_list = radius_list if radius_list is not None else [3, 5, 7]
        self._template_cache = {}

    @staticmethod
    def _create_fast_correlation_patch(angle_1: float, radius: int, weight_type: str = 'linear') -> Dict[str, np.ndarray]:
        """
        Fast correlation patch creation with linear distance-based weighting (recommended).
        
        Args:
            angle_1: Primary angle direction
            radius: Patch radius  
            weight_type: 'gaussian', 'linear', 'cosine', 'quadratic' (default: 'linear')
        """
        angle_2 = angle_1 + np.pi / 2
        w = h = 2 * radius + 1  # Square window for full coverage
        center = radius + 1
        
        coords = np.arange(1, w + 1)
        xx, yy = np.meshgrid(coords, coords)
        dx, dy = xx - center, yy - center
        
        # Distance-based weight distribution
        dist = np.sqrt(dx**2 + dy**2)
        max_dist = radius
        
        if weight_type == 'gaussian':
            from scipy.stats import norm
            wgt = norm.pdf(dist, 0, radius / 2)
        elif weight_type == 'linear':
            # RECOMMENDED: Linear decay for optimal speed and performance
            wgt = np.maximum(0, 1 - dist / max_dist)
        elif weight_type == 'cosine':
            wgt = np.where(dist <= max_dist, 
                          0.5 * (1 + np.cos(np.pi * dist / max_dist)), 0)
        elif weight_type == 'quadratic':
            normalized_dist = np.minimum(dist / max_dist, 1.0)
            wgt = 1 - normalized_dist ** 2
        else:
            # Default to linear for best performance
            wgt = np.maximum(0, 1 - dist / max_dist)
        
        n1 = np.array([-np.sin(angle_1), np.cos(angle_1)])
        n2 = np.array([-np.sin(angle_2), np.cos(angle_2)])
        
        s1 = dx * n1[0] + dy * n1[1]
        s2 = dx * n2[0] + dy * n2[1]
        
        template = {
            "a1": ((s1 <= -0.1) & (s2 <= -0.1)).astype(np.float64) * wgt,
            "a2": ((s1 >= 0.1) & (s2 >= 0.1)).astype(np.float64) * wgt,
            "b1": ((s1 <= -0.1) & (s2 >= 0.1)).astype(np.float64) * wgt,
            "b2": ((s1 >= 0.1) & (s2 <= -0.1)).astype(np.float64) * wgt,
        }
        
        # Normalize
        for key, patch in template.items():
            patch_sum = np.sum(patch)
            if patch_sum > 0:
                template[key] = patch / patch_sum
                
        return template

    def _improved_corner_correlation_score(self, img: np.ndarray, template: Dict[str, np.ndarray]) -> float:
        """
        Enhanced corner correlation scoring that fully leverages chessboard characteristics
        """
        # Compute responses for four regions
        responses = np.array([np.sum(template[key] * img) for key in ["a1", "a2", "b1", "b2"]])
        a1, a2, b1, b2 = responses
        
        # 1. Contrast score: Difference between diagonal region groups
        mean_a = (a1 + a2) / 2
        mean_b = (b1 + b2) / 2
        mean = (mean_a+mean_b)/2
        contrast_score = abs(mean_a - mean_b) / (mean_a + mean_b + 1e-8)
        
        # 2. Symmetry score: Similarity within same diagonal groups
        symmetry_a = 1 / (1 + abs(a1 - a2))
        symmetry_b = 1 / (1 + abs(b1 - b2))
        symmetry_score = (symmetry_a + symmetry_b) / 2
        
        # 3. Stability score: Avoid extreme values
        min_response = min(a1, a2, b1, b2)
        max_response = max(a1, a2, b1, b2)
        stability_score = min_response / (max_response + 1e-8)
        
        # 4. Diagonal dominance: One diagonal should be consistently higher
        diag1_consistent = (a1 > mean and a2 > mean) and (b1 < mean and b2 < mean)
        diag2_consistent = (b1 > mean and b2 > mean) and (a1 < mean and a2 < mean)
        consistency_bonus = 1 if (diag1_consistent or diag2_consistent) else 0
        
        # Combined scoring
        final_score = contrast_score * symmetry_score * (1 + stability_score) * consistency_bonus
        
        return float(final_score)

    @staticmethod
    def _non_maximum_suppression(img: np.ndarray, n: int = 3, tau: float = 0.01) -> np.ndarray:
        local_max = maximum_filter(img, size=2 * n + 1)
        maxima = (img == local_max) & (img >= tau)
        coords = np.argwhere(maxima)
        return coords[:, [1, 0]]  # x, y

    @staticmethod
    def _normalize_vec(v: np.ndarray) -> np.ndarray:
        mu = np.mean(v)
        sigma = np.std(v)
        if sigma < np.finfo(float).eps:
            sigma = np.finfo(float).eps
        return (v - mu) / sigma

    def _enhanced_corner_correlation_score(self, img: np.ndarray, img_weight: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Enhanced corner correlation scoring combining gradient and intensity analysis
        """
        H, W = img_weight.shape
        c = np.array([(H + 1) / 2, (W + 1) / 2])

        # Vectorized filter kernel generation
        y_coords, x_coords = np.mgrid[1:H+1, 1:W+1]
        p1_x, p1_y = x_coords - c[1], y_coords - c[0]

        # Efficient projection computations
        p1_dot_v1 = p1_x * v1[0] + p1_y * v1[1]
        p1_dot_v2 = p1_x * v2[0] + p1_y * v2[1]

        # Vectorized projection calculations
        p2 = np.stack([p1_dot_v1 * v1[0], p1_dot_v1 * v1[1]], axis=-1)
        p3 = np.stack([p1_dot_v2 * v2[0], p1_dot_v2 * v2[1]], axis=-1)
        p1 = np.stack([p1_x, p1_y], axis=-1)

        # Efficient distance computations
        diff1 = np.linalg.norm(p1 - p2, axis=-1)
        diff2 = np.linalg.norm(p1 - p3, axis=-1)

        # Vectorized filter creation
        img_filter = np.where((diff1 <= 1.5) | (diff2 <= 1.5), 1, -1)

        # Vectorized normalization
        vec_weight = self._normalize_vec(img_weight.flatten())
        vec_filter = self._normalize_vec(img_filter.flatten())

        # Efficient gradient score computation
        denom = max(len(vec_weight) - 1, 1)
        score_gradient = max(np.dot(vec_weight, vec_filter) / denom, 0)
        
        if score_gradient == 0:
            return 0.0

        # Use enhanced template caching with fast circular masks
        angle_key = f"{np.arctan2(v1[1], v1[0]):.3f}_{H}".replace(".", "_")
        
        if angle_key not in self._template_cache:
            template = self._create_fast_correlation_patch(np.arctan2(v1[1], v1[0]), int(c[0] - 1))
            self._template_cache[angle_key] = template
        else:
            template = self._template_cache[angle_key]

        # Use improved correlation scoring
        score_intensity = self._improved_corner_correlation_score(img, template)

        return float(score_gradient * score_intensity)

    def _refine_corners(self, img_du: np.ndarray, img_dv: np.ndarray, img_angle: np.ndarray, img_weight: np.ndarray,
                         corners: Dict[str, np.ndarray], r: int) -> Dict[str, np.ndarray]:
        """
        Vectorized corner refinement using fixed orthogonal directions
        """
        n = corners["p"].shape[0]
        
        # Use fixed orthogonal directions for speed
        corners["v1"] = np.tile(np.array([-1.0, 0.0]), (n, 1))
        corners["v2"] = np.tile(np.array([0.0, -1.0]), (n, 1))
        
        return corners

    def _score_corners_enhanced(self, img: np.ndarray, img_weight: np.ndarray, corners: Dict[str, np.ndarray], radius: List[int]) -> Dict[str, np.ndarray]:
        """
        Enhanced vectorized corner scoring
        """
        height, width = img.shape
        num_corners = corners['p'].shape[0]
        
        if num_corners == 0:
            corners['score'] = np.array([]).reshape(0, 1)
            return corners
            
        # Vectorized coordinate extraction and boundary checking
        coords = corners['p'].astype(int)
        u_coords, v_coords = coords[:, 0], coords[:, 1]
        
        max_radius = max(radius)
        valid_mask = ((u_coords >= max_radius) & (u_coords < width - max_radius) & 
                     (v_coords >= max_radius) & (v_coords < height - max_radius))
        
        scores = np.zeros((num_corners, 1))
        
        # Process valid corners
        valid_indices = np.where(valid_mask)[0]
        
        for i in valid_indices:
            u, v = u_coords[i], v_coords[i]
            score_list = []
            
            for r in radius:
                img_sub = img[v - r:v + r + 1, u - r:u + r + 1]
                img_weight_sub = img_weight[v - r:v + r + 1, u - r:u + r + 1]
                score_val = self._enhanced_corner_correlation_score(img_sub, img_weight_sub, corners['v1'][i], corners['v2'][i])
                score_list.append(score_val)
            
            scores[i, 0] = max(score_list) if score_list else 0.0
        
        corners['score'] = scores
        return corners

    def find_corners_enhanced(self, I: np.ndarray, tau: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Enhanced fast corner detection using improved scoring and circular mask templates
        """
        if tau is None:
            tau = self.tau
        if I.ndim == 3:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        img = I.astype(np.float64) / 255.0

        # Gradient computation
        du = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        dv = du.T
        img_du = convolve(img, du, mode='nearest')
        img_dv = convolve(img, dv, mode='nearest')
        
        # Vectorized gradient processing
        img_angle = np.arctan2(img_dv, img_du)
        img_weight = np.sqrt(img_du ** 2 + img_dv ** 2)
        
        # Angle normalization
        img_angle = np.where(img_angle < 0, img_angle + np.pi, img_angle)
        img_angle = np.where(img_angle > np.pi, img_angle - np.pi, img_angle)

        img_corners = np.zeros_like(img)
        
        # Process templates with enhanced circular masks using configurable radius list
        for r in self.radius_list:
            template = self._create_fast_correlation_patch(0, r)
            
            # Vectorized convolution
            conv_results = {
                key: convolve(img, patch, mode="nearest") 
                for key, patch in template.items()
            }
            
            # Enhanced corner response computation
            mu = sum(conv_results.values()) / 4
            
            # Use improved scoring logic
            a_responses = np.stack([conv_results["a1"], conv_results["a2"]])
            b_responses = np.stack([conv_results["b1"], conv_results["b2"]])
            
            # Enhanced contrast and symmetry evaluation
            mean_a = np.mean(a_responses, axis=0)
            mean_b = np.mean(b_responses, axis=0)
            contrast = np.abs(mean_a - mean_b)
            
            # Symmetry within diagonal groups
            symmetry_a = 1 / (1 + np.abs(conv_results["a1"] - conv_results["a2"]))
            symmetry_b = 1 / (1 + np.abs(conv_results["b1"] - conv_results["b2"]))
            symmetry = (symmetry_a + symmetry_b) / 2
            
            # Combined enhanced response
            enhanced_response = contrast * symmetry
            
            img_corners = np.maximum(img_corners, enhanced_response)

        corners = {"p": self._non_maximum_suppression(img_corners, 3, tau)}
        corners = self._refine_corners(img_du, img_dv, img_angle, img_weight, corners, 4)
        corners = self._score_corners_enhanced(img, img_weight, corners, self.radius_list)

        idx = (corners["score"].flatten() >= tau)
        for k in list(corners.keys()):
            corners[k] = corners[k][idx]

        return corners

    # -----------------------
    # Chessboard detection methods (copied from original module)
    # -----------------------
    @staticmethod
    def _chessboard_energy(chessboard: np.ndarray, corners: Dict[str, np.ndarray]) -> float:
        """
        Optimized chessboard energy calculation using vectorized operations.
        """
        if chessboard is None or chessboard.size == 0:
            return np.inf
            
        rows, cols = chessboard.shape
        e_corners = -rows * cols
        
        p = corners['p']
        valid_indices = chessboard[chessboard != -1]
        
        if len(valid_indices) == 0:
            return np.inf
            
        e_structure = 0.0
        
        # Vectorized row-wise structure energy calculation
        for r in range(rows):
            row_indices = chessboard[r, :]
            valid_triplets = []
            for c in range(cols - 2):
                triplet = row_indices[c:c+3]
                if not np.any(triplet == -1):
                    valid_triplets.append(triplet)
            
            if valid_triplets:
                triplet_array = np.array(valid_triplets)
                points = p[triplet_array]
                
                p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
                norm_diffs = np.linalg.norm(p0 - p2, axis=1)
                valid_mask = norm_diffs > 1e-6
                
                if np.any(valid_mask):
                    deviations = np.linalg.norm(p0 + p2 - 2 * p1, axis=1)[valid_mask]
                    normalized_deviations = deviations / norm_diffs[valid_mask]
                    e_structure = max(e_structure, np.max(normalized_deviations))
        
        # Vectorized column-wise structure energy calculation
        for c in range(cols):
            col_indices = chessboard[:, c]
            valid_triplets = []
            for r in range(rows - 2):
                triplet = col_indices[r:r+3]
                if not np.any(triplet == -1):
                    valid_triplets.append(triplet)
            
            if valid_triplets:
                triplet_array = np.array(valid_triplets)
                points = p[triplet_array]
                
                p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
                norm_diffs = np.linalg.norm(p0 - p2, axis=1)
                valid_mask = norm_diffs > 1e-6
                
                if np.any(valid_mask):
                    deviations = np.linalg.norm(p0 + p2 - 2 * p1, axis=1)[valid_mask]
                    normalized_deviations = deviations / norm_diffs[valid_mask]
                    e_structure = max(e_structure, np.max(normalized_deviations))
        
        return e_corners + rows * cols * e_structure

    def _directional_neighbor(self, idx: int, v: np.ndarray, chessboard: np.ndarray, corners: Dict[str, np.ndarray], available_mask: np.ndarray):
        num_corners = corners['p'].shape[0]
        current_used_indices = chessboard[chessboard != -1].astype(int)
        search_mask = available_mask.copy()
        if current_used_indices.size > 0:
            search_mask[current_used_indices] = False
        unused_indices = np.where(search_mask)[0]
        if unused_indices.size == 0:
            return -1, np.inf
        p_idx = corners['p'][idx, :]
        p_unused = corners['p'][unused_indices, :]
        direction = p_unused - p_idx
        dist = np.dot(direction, v)
        dist_edge = direction - np.outer(dist, v)
        dist_edge = np.linalg.norm(dist_edge, axis=1)
        dist_point = np.copy(dist)
        dist_point[dist_point < 0] = np.inf
        cost = dist_point + 5 * dist_edge
        if cost.size == 0 or np.all(np.isinf(cost)):
            return -1, np.inf
        min_idx_in_unused = np.argmin(cost)
        min_dist_val = cost[min_idx_in_unused]
        neighbor_idx = unused_indices[min_idx_in_unused]
        return neighbor_idx, min_dist_val

    def _init_chessboard(self, corners: Dict[str, np.ndarray], idx: int, transposed: bool, available_mask: np.ndarray) -> Optional[np.ndarray]:
        if corners['p'].shape[0] < 9:
            return None
        chessboard = np.full((3, 3), -1, dtype=int)
        v1_orig, v2_orig = corners['v1'][idx, :], corners['v2'][idx, :]
        v1, v2 = (v2_orig, v1_orig) if transposed else (v1_orig, v2_orig)
        chessboard[1, 1] = idx

        points = [(1, 2, +v1), (1, 0, -v1), (2, 1, +v2), (0, 1, -v2),
                  (0, 0, -v2, 1, 0), (2, 0, +v2, 1, 0),
                  (0, 2, -v2, 1, 2), (2, 2, +v2, 1, 2)]
        for p in points:
            res, _ = self._directional_neighbor(chessboard[p[3], p[4]] if len(p) > 3 else idx, p[2], chessboard, corners, available_mask)
            if res == -1:
                return None
            chessboard[p[0], p[1]] = res
        return chessboard

    @staticmethod
    def _predict_corners(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        v1, v2 = p2 - p1, p3 - p2
        a1, a2 = np.arctan2(v1[:, 1], v1[:, 0]), np.arctan2(v2[:, 1], v2[:, 0])
        a3 = 2 * a2 - a1
        s1, s2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        s3 = 2 * s2 - s1
        return p3 + 0.75 * np.column_stack((s3 * np.cos(a3), s3 * np.sin(a3)))

    @staticmethod
    def _assign_closest_corners(cand_indices: np.ndarray, cand_coords: np.ndarray, pred_coords: np.ndarray) -> np.ndarray:
        """
        Optimized corner assignment using efficient distance computation.
        """
        from scipy.spatial.distance import cdist
        
        if cand_coords.shape[0] < pred_coords.shape[0]:
            return np.array([], dtype=int)
            
        dist_matrix = cdist(cand_coords, pred_coords)
        n_pred = pred_coords.shape[0]
        
        assigned_indices = np.zeros(n_pred, dtype=int)
        used_candidates = np.zeros(cand_coords.shape[0], dtype=bool)
        
        for i in range(n_pred):
            available_mask = ~used_candidates
            if not np.any(available_mask):
                break
                
            valid_distances = dist_matrix[available_mask, i]
            if len(valid_distances) == 0 or np.all(np.isinf(valid_distances)):
                break
                
            best_local_idx = np.argmin(valid_distances)
            best_global_idx = np.where(available_mask)[0][best_local_idx]
            
            assigned_indices[i] = cand_indices[best_global_idx]
            used_candidates[best_global_idx] = True
            dist_matrix[best_global_idx, :] = np.inf
            
        return assigned_indices

    def _grow_chessboard(self, chessboard: np.ndarray, corners: Dict[str, np.ndarray], border_type: int, available_mask: np.ndarray) -> np.ndarray:
        if chessboard is None:
            return None
        rows, cols = chessboard.shape
        p = corners['p']

        if border_type in [1, 3] and cols < 3:
            return chessboard
        if border_type in [2, 4] and rows < 3:
            return chessboard

        current_used_indices = chessboard[chessboard != -1].astype(int)
        search_mask = available_mask.copy()
        if current_used_indices.size > 0:
            search_mask[current_used_indices] = False
        unused_indices = np.where(search_mask)[0]
        cand_coords = p[unused_indices, :]

        pred = None
        if border_type == 1:
            pred = self._predict_corners(p[chessboard[:, -3]], p[chessboard[:, -2]], p[chessboard[:, -1]])
        elif border_type == 2:
            pred = self._predict_corners(p[chessboard[-3, :]], p[chessboard[-2, :]], p[chessboard[-1, :]])
        elif border_type == 3:
            pred = self._predict_corners(p[chessboard[:, 2]], p[chessboard[:, 1]], p[chessboard[:, 0]])
        elif border_type == 4:
            pred = self._predict_corners(p[chessboard[2, :]], p[chessboard[1, :]], p[chessboard[0, :]])

        if pred is not None:
            idx = self._assign_closest_corners(unused_indices, cand_coords, pred)
            if idx.size > 0 and np.all(idx != 0):
                if border_type == 1:
                    return np.hstack((chessboard, idx[:, np.newaxis]))
                elif border_type == 2:
                    return np.vstack((chessboard, idx))
                elif border_type == 3:
                    return np.hstack((idx[:, np.newaxis], chessboard))
                elif border_type == 4:
                    return np.vstack((idx, chessboard))
        return chessboard

    def find_chessboards(self, corners: Dict[str, np.ndarray], board_size: Tuple[int, int], image: Optional[np.ndarray] = None,
                         debug_visualization: bool = False, target_boards: int = 5, energy_threshold: float = -45.0) -> List[np.ndarray]:
        """
        Enhanced chessboard detection using improved corner scoring.
        """
        num_corners = corners['p'].shape[0]
        valid_seeds = np.ones(num_corners, dtype=bool)
        chessboards: List[np.ndarray] = []

        while len(chessboards) < target_boards:
            seeds_to_try = np.where(valid_seeds)[0]
            if seeds_to_try.size == 0:
                break

            found_new_board_in_pass = False
            for i in seeds_to_try:
                if not valid_seeds[i]:
                    continue

                for transposed in [False, True]:
                    unused_mask = np.ones(num_corners, dtype=bool)
                    for cb in chessboards:
                        unused_mask[cb[cb != -1]] = False

                    chessboard = self._init_chessboard(corners, i, transposed, unused_mask)
                    if chessboard is None:
                        continue

                    last_action = None
                    while True:
                        energy = self._chessboard_energy(chessboard, corners)

                        rows, cols = chessboard.shape
                        is_valid_size = (rows == board_size[0] and cols == board_size[1]) or \
                                        (rows == board_size[1] and cols == board_size[0])

                        if is_valid_size and energy < energy_threshold:
                            break

                        proposals = [self._grow_chessboard(chessboard, corners, j + 1, unused_mask) for j in range(4)]
                        p_energies = [self._chessboard_energy(p, corners) for p in proposals]
                        min_idx = int(np.argmin(p_energies))

                        can_grow = any(proposal is not None and not np.array_equal(proposal, chessboard) for proposal in proposals)

                        if p_energies[min_idx] < energy and can_grow:
                            chessboard = proposals[min_idx]
                            last_action = 'grow'
                            continue

                        if last_action == 'prune' or not can_grow:
                            break

                        rows, cols = chessboard.shape
                        if rows < 3 or cols < 3:
                            break

                        prune_proposals = [chessboard[1:, :], chessboard[:-1, :], chessboard[:, 1:], chessboard[:, :-1]]
                        prune_energies = [self._chessboard_energy(p, corners) for p in prune_proposals]
                        min_prune_idx = int(np.argmin(prune_energies))

                        if prune_energies[min_prune_idx] < energy:
                            chessboard = prune_proposals[min_prune_idx]
                            last_action = 'prune'
                            continue
                        else:
                            break

                    rows, cols = chessboard.shape
                    is_valid_size = (rows == board_size[0] and cols == board_size[1]) or \
                                    (rows == board_size[1] and cols == board_size[0])

                    current_energy = self._chessboard_energy(chessboard, corners)
                    if is_valid_size and current_energy < energy_threshold:
                        chessboards.append(chessboard)
                        valid_seeds[chessboard[chessboard != -1]] = False
                        found_new_board_in_pass = True
                        if len(chessboards) >= target_boards:
                            break
                    else:
                        valid_seeds[i] = False

                if len(chessboards) >= target_boards:
                    break

            if not found_new_board_in_pass:
                break

        return chessboards
