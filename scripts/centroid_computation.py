import cv2 as cv
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.algorithms import polylabel


def compute_3d_position_from_mask_and_depth(depth_image, person_mask, camera_intrinsics, depth_scale=1, depth_filter_percentage=0.2):
    """
    This function calculates the 3D position (X, Y, Z) of a person from the depth image and its corresponding mask.
    It calculates the 3D position of a person by first finding a valid point inside the mask (centroid or furthest point)
    and then using depth information.
    
    Parameters:
    - depth_image: Depth image where each pixel value represents the depth (in meters)
    - person_mask: Binary mask where the person's region is marked
    - camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy)
    - depth_scale: Scale factor to convert depth to meters
    - depth_filter_percentage: Fraction of the image size to use for depth filtering
    
    Returns:
    - cx, cy: Centroid coordinates
    - X, Y, Z: 3D position of the person in the world coordinate system
    """
    # Camera intrinsic parameters
    K = np.array(camera_intrinsics["K"])
    fx_depth = K[0, 0]
    fy_depth = K[1, 1]
    cx_depth = K[0, 2]
    cy_depth = K[1, 2]

    # 1. Get Bounding Box Around the Mask
    bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box_from_mask(person_mask)

    # 2. Optionally, expand the bounding box (if you want a larger region)
    # bbox_x, bbox_y, bbox_w, bbox_h = expand_bounding_box(bbox_x, bbox_y, bbox_w, bbox_h, depth_image.shape)

    # 3. Get centroid or furthest point (centroid is used here)
    cx, cy = get_centroid_of_mask(person_mask)

    if cx is None or person_mask[int(cy), int(cx), 0] != 255:
        cx, cy = get_furthest_point_from_mask_edge(person_mask)

    # 4. Compute depth for the chosen point (centroid or furthest point)
    depth = get_median_depth(depth_filter_percentage, int(cy), int(cx), depth_image.shape[0], depth_image.shape[1], depth_image, person_mask)

    # 5. If depth is valid, convert it to 3D coordinates
    if depth > 0:
        z = depth / depth_scale
        x = (int(cx) - cx_depth) * z / fx_depth
        y = (int(cy) - cy_depth) * z / fy_depth
        return cx, cy, x, y, z

    # If no valid depth or centroid found, return None for 3D position
    return None, None, None, None, None


def get_bounding_box_from_mask(mask):
    """
    Computes the bounding box around the person's mask using contours.
    
    Parameters:
    - mask: Binary mask image
    
    Returns:
    - x, y, w, h: Bounding box coordinates and size
    """
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, None, None

    # Get the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)
    return x, y, w, h


def expand_bounding_box(x, y, w, h, image_shape, expand_ratio=0.1):
    """
    Expands the bounding box by a specified ratio to get more context around the person.
    
    Parameters:
    - x, y, w, h: Bounding box coordinates and size
    - image_shape: Shape of the image (height, width)
    - expand_ratio: Ratio by which to expand the bounding box
    
    Returns:
    - new_x, new_y, new_w, new_h: Expanded bounding box coordinates and size
    """
    height, width = image_shape
    expand_x = int(w * expand_ratio)
    expand_y = int(h * expand_ratio)

    new_x = max(0, x - expand_x)
    new_y = max(0, y - expand_y)
    new_w = min(width, x + w + expand_x) - new_x
    new_h = min(height, y + h + expand_y) - new_y

    return new_x, new_y, new_w, new_h


def get_centroid_of_mask(mask):
    """
    Computes the centroid of the largest contour in the mask using polylabel.
    
    Parameters:
    - mask: Binary mask image
    
    Returns:
    - cx, cy: Centroid coordinates of the largest contour in the mask
    """
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    polygon = Polygon(largest_contour.squeeze())
    
    try:
        centroid = polylabel.polylabel(polygon, tolerance=1.0)
    except:
        return None, None
    
    return centroid.x, centroid.y


def get_furthest_point_from_mask_edge(mask):
    """Finds the furthest point from the edge of the mask."""
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    distance_transform = cv.distanceTransform(mask_gray, distanceType=cv.DIST_L2, maskSize=5).astype(np.float32)
    cy, cx = np.where(distance_transform == distance_transform.max())
    return cx[0], cy[0]


def get_median_depth(depth_filter_percentage, cy, cx, height, width, depth_image, mask):
    """Calculates the median depth within a rectangular region around the given (cy, cx) coordinates."""
    half_height = int(height * depth_filter_percentage)
    half_width = int(width * depth_filter_percentage)

    y0 = max(0, cy - half_height)
    y1 = min(height, cy + half_height + 1)
    x0 = max(0, cx - half_width)
    x1 = min(width, cx + half_width + 1)

    depth_region = depth_image[y0:y1, x0:x1].flatten()
    if mask is not None:
        mask_region = mask[y0:y1, x0:x1].flatten()
        valid_depths = [d for i, d in enumerate(depth_region) if mask_region[i] == 255]
    else:
        valid_depths = depth_region
    
    if len(valid_depths) > 0:
        depth = np.median(valid_depths)
        if math.isnan(depth):
            depth = 0
    else:
        depth = 0
    
    return depth
