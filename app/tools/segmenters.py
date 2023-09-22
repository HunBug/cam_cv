from enum import Enum
import cv2
import numpy as np


class AveragingMethod(Enum):
    WEIGHTED_MEAN = 0
    ALL = 1


def segment_by_homography(frame1, frame2) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)

    matched_keypoints1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    matched_keypoints2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )

    # calculate homography matrix
    homography_matrix1, inliers = cv2.findHomography(
        matched_keypoints1, matched_keypoints2, cv2.RANSAC, 5.0
    )
    inliers_percentage = float(inliers.sum()) / float(len(inliers))
    if inliers_percentage < 0.5:
        print("Can't find homography matrix.")
        return None, None

    # warp perspective
    rectified_image1 = cv2.warpPerspective(
        frame1, homography_matrix1, (frame1.shape[1], frame1.shape[0])
    )

    # calculate disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=31)
    gray_rectified_image1 = cv2.cvtColor(rectified_image1, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray2, gray_rectified_image1)

    DISPARITY_THRESHOLD = 0
    # background_mask = (disparity < DISPARITY_THRESHOLD).astype(np.uint8)
    foreground_mask = np.where(disparity >= DISPARITY_THRESHOLD)
    background_mask = np.where(disparity < DISPARITY_THRESHOLD)

    foreground = frame2.copy()
    foreground[foreground_mask] = (0, 0, 0)
    background = frame2.copy()
    background[background_mask] = (0, 0, 0)

    return foreground, background


def segment_by_optical_flow(frame1, frame2) -> tuple[np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # create histogram of optical flow
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # find the median of the magnitude
    threshold = np.median(mag)
    

    # Create a mask by thresholding the magnitude
    mask = mag > threshold

    # Create a mask to remove the background
    background_mask = np.where(mask == 0)
    foreground_mask = np.where(mask == 1)

    foreground = frame2.copy()
    foreground[foreground_mask] = (0, 0, 0)
    background = frame2.copy()
    background[background_mask] = (0, 0, 0)

    return foreground, background


def segmentation_refinement(img: np.ndarray) -> np.ndarray:
    """Refine the segmentation mask using morphological operations."""
    # check if the input image is grayscale
    if len(img.shape) == 3:
        input_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # make sure the the image is uint8
        input_mask = img.copy().astype(np.uint8)
    input_mask = cv2.threshold(input_mask, 96, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    # get the largest connected component
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(labels)
    mask[labels == largest_label] = 255

    return mask


def average_segmentation_over_time(
    images: list[np.ndarray], mode: AveragingMethod = AveragingMethod.WEIGHTED_MEAN
) -> np.ndarray:
    # weighted average of the segmentation masks
    # the middle frames are weighted more than the first and last frames
    input_masks = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    input_masks = [
        cv2.threshold(input_mask, 1, 255, cv2.THRESH_BINARY)[1]
        for input_mask in input_masks
    ]

    if mode == AveragingMethod.WEIGHTED_MEAN:
        # weights
        weights = np.linspace(0, 1, len(images))
        weights = np.concatenate((weights, weights[::-1]))

        # normalize weights
        weights = weights / np.sum(weights)

        # average
        output_mask = np.zeros_like(input_masks[0], dtype=np.float32)
        for input_mask, weight in zip(input_masks, weights):
            output_mask += input_mask * weight

        return output_mask.astype(np.uint8)
    elif mode == AveragingMethod.ALL:
        # use and operation to get the pixels that are white in all images
        output_mask = np.ones_like(input_masks[0], dtype=np.uint8) * 255
        for input_mask in input_masks:
            output_mask = cv2.bitwise_and(output_mask, input_mask)

        return output_mask
    raise ValueError("Invalid averaging method")