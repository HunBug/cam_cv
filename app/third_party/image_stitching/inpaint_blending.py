import cv2
import numpy as np

from app.third_party.image_stitching.image import Image
from app.third_party.image_stitching.utils import get_new_parameters, single_weights_matrix


def add_image(
    panorama: np.ndarray, image: Image, offset: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add a new image to the panorama using the provided offset and weights.

    Args:
        panorama: Existing panorama
        image: Image to add to the panorama
        offset: Offset already applied to the panorama
        weights: Weights matrix of the panorama

    Returns:
        panorama: Panorama with the new image
        offset: New offset matrix
        weights: New weights matrix
    """
    H = offset @ image.H
    size, added_offset = get_new_parameters(panorama, image.image, H)

    new_image = cv2.warpPerspective(image.image, added_offset @ H, size)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros_like(new_image)
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size)
        weights = cv2.warpPerspective(weights, added_offset, size)

    weights = np.ones_like(panorama)
    image_weights = single_weights_matrix(image.image.shape)
    image_weights = np.ones(image.image.shape[:2])
    image_weights = np.repeat(
        cv2.warpPerspective(image_weights, added_offset @ H, size)[:, :, np.newaxis],
        3,
        axis=2,
    )

    # # Define the threshold values
    low_thresh = 50
    high_thresh = 200
    pav = np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) 
    niav = np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2)
    mask1 = np.logical_and(pav > low_thresh, pav < high_thresh)
    mask2 = np.logical_and(niav > low_thresh, niav < high_thresh)
    common_mask = np.logical_and(mask1, mask2)

    # Calculate the average of the pixels within the threshold range
    avg1 = np.mean(panorama[common_mask])
    avg2 = np.mean(new_image[common_mask])

    # Calculate the gain that makes the average of the two images the same
    if avg2 == 0:
        gain = 1
    elif avg1 == 0:
        gain = 1
    else:
        gain = avg1 / avg2
    if np.isnan(gain):
        gain = 1

    # Apply the gain to the second image
    panorama = panorama / gain

    panorama = np.maximum(panorama, new_image)

    new_weights = (weights + image_weights) / (weights + image_weights).max()

    return panorama, added_offset @ offset, new_weights


def inpaint_blending(images: list[Image]) -> np.ndarray:
    """
    Build a panorama from the given images using simple blending.

    Args:
        images: Images to build the panorama from

    Returns:
        panorama: Panorama of the given images
    """
    panorama = None
    weights = None
    offset = np.eye(3)
    for image in images:
        panorama, offset, weights = add_image(panorama, image, offset, weights)

    return panorama
