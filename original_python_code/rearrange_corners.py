import numpy as np
import cv2

def _is_upper_left_black(cb, corner_coords, gray_image):
    """
    Helper function to check if the upper-left square of the board is black.
    Python port of isUpperLeftBlack.m
    """
    # Calculate the center of the top-left square
    ul_center_x = (corner_coords[cb[0, 0], 0] + corner_coords[cb[1, 1], 0]) / 2
    ul_center_y = (corner_coords[cb[0, 0], 1] + corner_coords[cb[1, 1], 1]) / 2

    # Calculate the center of the adjacent square to the right
    next_center_x = (corner_coords[cb[0, 1], 0] + corner_coords[cb[1, 2], 0]) / 2
    next_center_y = (corner_coords[cb[0, 1], 1] + corner_coords[cb[1, 2], 1]) / 2

    # Sample a 5x5 patch around each center and get the mean intensity
    h, w = gray_image.shape
    
    def get_mean_intensity(x, y):
        # Ensure coordinates are within image bounds
        y_start = max(int(np.ceil(y)) - 2, 0)
        y_end = min(int(np.ceil(y)) + 3, h)
        x_start = max(int(np.ceil(x)) - 2, 0)
        x_end = min(int(np.ceil(x)) + 3, w)
        patch = gray_image[y_start:y_end, x_start:x_end]
        return np.mean(patch)

    ul_mean = get_mean_intensity(ul_center_x, ul_center_y)
    next_mean = get_mean_intensity(next_center_x, next_center_y)

    # If the upper-left square is darker, it's considered black
    return ul_mean < next_mean

def rearrange_corners(chessboards, corners, image):
    """
    Rearranges the corners of each chessboard to a canonical orientation.
    Python port of rearrangeCorners.m
    """
    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    image_points_cell = []
    for cb in chessboards:
        # Make the board wider than it is tall
        if cb.shape[0] > cb.shape[1]:
            cb = np.rot90(cb, k=1)

        # Orient the board so that (0,0) is on a black square
        if not _is_upper_left_black(cb, corners['p'], gray_image):
            cb = np.rot90(cb, k=2)
        
        # Flatten the corner indices in column-major order (like MATLAB)
        # and get the corresponding coordinates
        rearranged_indices = cb.flatten('F')
        image_points_cell.append(corners['p'][rearranged_indices, :])
        
    return image_points_cell
