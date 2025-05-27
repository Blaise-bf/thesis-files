from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.signal import convolve2d

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Resize image while maintaining aspect ratio
def resize_to_square(
    img: np.ndarray,
    target_size: int,
    pad_color: int = 0,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resizes an image to a square while maintaining aspect ratio.
    Pads the remaining space with `pad_color` (default: 0/black).

    Args:
        img (np.ndarray): Input image (grayscale or color).
        target_size (int): Desired width & height of the square output.
        pad_color (int): Padding color (default 0 for black).

    Returns:
        np.ndarray: Square image of size (target_size, target_size).
    """
    h, w = img.shape[:2]

    # Scale the image to fit inside the target square
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation )

    # Calculate padding to center the image
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Apply padding
    square_img = cv2.copyMakeBorder(
        resized_img,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return square_img


def read_image_frompath(
    mask_paths: list[str],
    size: int | None = None,
    pad_color: int = 0,
    interpolation: int = cv2.INTER_CUBIC
) -> list[np.ndarray]:
    """
    Reads images from paths and optionally resizes them to squares.

    Args:
        mask_paths (list[str]): List of image file paths.
        size (int | None): If provided, resize images to (size x size).
        pad_color (int): Padding color (default 0 for black).

    Returns:
        list[np.ndarray]: List of images (resized if `size` is given).
    """
    images = []
    for path in mask_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image at {path}")
            continue

        if size is not None:
            img = resize_to_square(img, size, pad_color, interpolation)
        images.append(img)

    return images


def load_base_unetmodel():
    model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",  # EfficientNet-b4 encoder
    encoder_weights="imagenet",      # Pretrained weights
    in_channels=1,                   # RGB channels
    classes=1                       # Single binary mask class
)
    return model

def vertical_distance_features(atrium_mask, catheter_mask):
    try:
        # Convert PyTorch tensors to numpy if needed
        if hasattr(atrium_mask, 'cpu'):
            atrium_mask = atrium_mask.cpu().numpy()
        if hasattr(catheter_mask, 'cpu'):
            catheter_mask = catheter_mask.cpu().numpy()

        # 1) Find atrial boundaries
        atrium_rows = np.nonzero(atrium_mask)[0]
        if len(atrium_rows) == 0:
            raise ValueError("Empty atrium mask")
        atrium_top, atrium_bottom = atrium_rows.min(), atrium_rows.max()

        # 2) Process catheter mask
        cath_mask = catheter_mask.astype(bool)
        if not np.any(cath_mask):
            raise ValueError("Empty catheter mask")

        # 3) Find catheter skeleton and endpoints
        skeleton = skeletonize(cath_mask)

        # Alternative endpoint detection without thin()
        # Endpoints are points with exactly one neighbor in skeleton
        kernel = np.array([[1,1,1],
                           [1,0,1],
                           [1,1,1]])
        conv = convolve2d(skeleton.astype(int), kernel, mode='same')
        endpoints = np.argwhere((skeleton > 0) & (conv == 1))

        # 4) Calculate thickness map using distance transform
        dt_cat = distance_transform_edt(cath_mask)
        thickness_map = dt_cat * 2  # Convert radius to diameter

        # 5) Identify potential tip candidates (endpoints + thinnest points)
        if len(endpoints) == 0:  # Fallback for looped catheters
            endpoints = np.argwhere(skeleton)

        # Find thinnest endpoint
        min_thickness = np.inf
        tip_row = None
        for y, x in endpoints:
            thickness = thickness_map[y, x]
            if thickness < min_thickness:
                min_thickness = thickness
                tip_row = y

        # 6) Fallback if no endpoints found
        if tip_row is None:
            cath_rows = np.nonzero(cath_mask)[0]
            cath_top, cath_bottom = cath_rows.min(), cath_rows.max()
            tip_row = cath_top if (thickness_map[cath_top].mean() <
                                 thickness_map[cath_bottom].mean()) else cath_bottom

        # 7) Calculate distances relative to tip
        dist_to_top = tip_row - atrium_top
        dist_to_bottom = atrium_bottom - tip_row

        return {
            "atrium_top": int(atrium_top),
            "atrium_bottom": int(atrium_bottom),
            "tip_row": int(tip_row),
            "dist_to_atria_top": int(dist_to_top),
            "dist_to_atria_bottom": int(dist_to_bottom),
            "min_thickness": float(min_thickness)
        }

    except ValueError as e:
        return {
            "atrium_top": np.nan,
            "atrium_bottom": np.nan,
            "tip_row": np.nan,
            "dist_to_atria_top": np.nan,
            "dist_to_atria_bottom": np.nan,
            "min_thickness": np.nan
        }

def hu_features(mask):
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log

def extract_features(atrium_mask, catheter_mask, index=None):
    atrium_mask = atrium_mask.cpu().numpy()
    catheter_mask = catheter_mask.cpu().numpy()
    # image = image.cpu().numpy() if image is not None else None

    feats = {}

    try:
        A = regionprops(atrium_mask.astype(int))[0]
        C = regionprops(catheter_mask.astype(int))[0]
        minr, minc, maxr, maxc = A.bbox
        H = maxr - minr

        r_cent, c_cent = C.centroid
        feats["loc_norm"] = (r_cent - minr) / H

        third = H // 3
        upper_mask = np.zeros_like(atrium_mask)
        upper_mask[minr:minr + 2 * third, minc:maxc] = 1
        feats["frac_upper"] = (
            np.logical_and(catheter_mask, upper_mask).sum() / catheter_mask.sum()
        )

        dt = distance_transform_edt(atrium_mask)
        catheter_mask_bool = catheter_mask.astype(bool)
        dvals = dt[catheter_mask_bool]
        feats.update({
            "d_mean": dvals.mean(),
            "d_std": dvals.std(),
            "d_min": dvals.min(),
            "d_max": dvals.max(),
        })

        feats.update({
            "length": C.major_axis_length,
            "orientation": C.orientation,
            "eccentricity": C.eccentricity
        })

        skel = skeletonize(catheter_mask)
        ys, xs = np.nonzero(skel)
        vecs = np.diff(np.vstack([ys, xs]).T, axis=0)
        if vecs.shape[0] > 1:
            angles = np.arctan2(vecs[:, 0], vecs[:, 1])
            feats["curvature"] = np.std(np.diff(angles))
        else:
            feats["curvature"] = np.nan

        hu_log_catheter = hu_features(catheter_mask)
        hu_log_atrium = hu_features(atrium_mask)
        feats.update({f"hu_catheter_{i}": v for i, v in enumerate(hu_log_catheter)})
        feats.update({f"hu_atrium_{i}": v for i, v in enumerate(hu_log_atrium)})

        intersection = np.logical_and(catheter_mask, atrium_mask).sum()
        union = np.logical_or(catheter_mask, atrium_mask).sum()
        feats.update({
            "frac_catheter_in_atrium": intersection / catheter_mask.sum(),
            "frac_atrium_covered": intersection / atrium_mask.sum(),
            "iou": intersection / union,
        })

        feats.update(vertical_distance_features(atrium_mask, catheter_mask))



        return feats

    except IndexError as e:
        print(f"IndexError at index {index}: {e}")
    except Exception as e:
        print(f"Unexpected error at index {index}: {e}")

    # Fallback: return NaNs for all features
    fallback_feats = {k: np.nan for k in feats.keys()}
    fallback_feats.update({f"hu_catheter_{i}": np.nan for i in range(7)})
    fallback_feats.update({f"hu_atrium_{i}": np.nan for i in range(7)})
    fallback_feats.update({
        "loc_norm": np.nan,
        "frac_upper": np.nan,
        "d_mean": np.nan,
        "d_std": np.nan,
        "d_min": np.nan,
        "d_max": np.nan,
        "length": np.nan,
        "orientation": np.nan,
        "eccentricity": np.nan,
        "curvature": np.nan,
        "frac_catheter_in_atrium": np.nan,
        "frac_atrium_covered": np.nan,
        "iou": np.nan,
        "atrium_top": np.nan,
        "atrium_bottom": np.nan,
        "tip_row": np.nan,
        "dist_to_atria_top": np.nan,
        "dist_to_atria_bottom": np.nan,
        "min_thickness": np.nan
    })
    return fallback_feats
