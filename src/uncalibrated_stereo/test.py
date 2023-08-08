import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
from pathlib import Path
import json

import sys

sys.path.append("../")
import camera_calibration as cc

# make sure to reload the module if it was already imported
import importlib

importlib.reload(cc)


def triangulate_points(P1, P2, pts1, pts2):
    # Convert points to homogeneous coordinates
    pts1_homog = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)
    pts2_homog = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)

    # Compute the cross-product matrix for each point
    x1 = np.cross(pts1_homog, np.tile(P1.T[2, :], (pts1_homog.shape[0], 1)))
    x2 = np.cross(pts2_homog, np.tile(P2.T[2, :], (pts2_homog.shape[0], 1)))

    # Stack the cross-product matrices into a single matrix
    A = np.concatenate((x1[:, :3], x2[:, :3]), axis=1)

    # Compute the SVD of the matrix A
    _, _, V = np.linalg.svd(A)

    # Extract the last column of V and normalize it
    X_homog = V[-1, :]
    X_homog /= X_homog[-1]

    # Convert the homogeneous coordinates to 3D coordinates
    X = X_homog[:3]

    return X


def DLT(P1, P2, point1, point2):
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    _, _, V = np.linalg.svd(B)

    return V[3, 0:3] / V[3, 3]


DATA_ROOT = Path("./data")
IMAGE_1_NAME = Path("left.jpg")
IMAGE_2_NAME = Path("right.jpg")
# IMAGE_1_NAME = Path("tsukuba_l.png")
# IMAGE_2_NAME = Path("tsukuba_r.png")
CALIBRATION_FILE = Path("calibration.json")

# load the images
img1 = cv2.imread(str(DATA_ROOT / IMAGE_1_NAME))
img2 = cv2.imread(str(DATA_ROOT / IMAGE_2_NAME))

# load the camera matrices
camera_calibration = cc.CameraCalibParams.from_dict(
    json.load(open(str(DATA_ROOT / CALIBRATION_FILE)))
)

print(img1.shape)
print(img2.shape)
print(camera_calibration)

K = camera_calibration.camera_matrix

# take two images and find the fundamental matrix between them
# use the fundamental matrix to find the epipolar lines in the second image
# use the epipolar lines to find the epipoles
# use the epipoles to find the essential matrix
# use the essential matrix to find the camera matrices
# use the camera matrices to find the 3D points

# find the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)  # left image
kp2, des2 = sift.detectAndCompute(img2, None)  # right image

# find the matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.25 * n.distance:
        good.append([m])
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

# find the fundamental matrix
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
# find the epilines
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# # find the epipoles
# epipole1 = cv2.computeCorrespondEpilines(np.array([[0,0]]), 2,F)
# epipole1 = epipole1.reshape(-1,3)
# epipole2 = cv2.computeCorrespondEpilines(np.array([[0,0]]), 1,F)
# epipole2 = epipole2.reshape(-1,3)


# Compute the essential matrix from the fundamental matrix and the camera matrix
E = np.matmul(np.matmul(K.T, F), K)

# Decompose the essential matrix to get the rotation and translation
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Form the camera projection matrices
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))


# triangulated_points = triangulate_points(P1, P2, pts1, pts2)
# points1 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

# List to store the triangulated 3D coordinates for each pair of points
triangulated_3d_points = []

for i in range(pts1.shape[0]):
    # Get the i-th pair of points and corresponding camera projection matrices
    pt1 = pts1[i]
    pt2 = pts2[i]
    P1_i = P1
    P2_i = P2

    # Triangulate the i-th pair of points
    # point_3d = triangulate_points(P1_i, P2_i, pt1.reshape(1, 2), pt2.reshape(1, 2))
    point_3d = DLT(P1, P2, pt1, pt2)

    # Append the triangulated point to the list
    triangulated_3d_points.append(point_3d)

# Convert the list to a NumPy array
triangulated_3d_points = np.array(triangulated_3d_points)

# convert to inhomogeneous coordinates
# points1 = points1/points1[3,:]
# points2 = points2/points2[3,:]
# points3 = points3/points3[3,:]
# points4 = points4/points4[3,:]

# plot the points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(triangulated_3d_points[:, 0], triangulated_3d_points[:, 1], triangulated_3d_points[:, 2])
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()


# Compute rectification transforms and projection matrices
ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, E, img1.shape[:2])

# Rectify images
img1_rectified = cv2.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
img2_rectified = cv2.warpPerspective(img2, H2, (img1.shape[1], img1.shape[0]))

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(img1_rectified)
ax[1].imshow(img2_rectified)
plt.show()

# Calculate disparity map (example: block matching)
window_size = 10
min_disparity = 0
num_disparities = 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity, numDisparities=num_disparities, blockSize=window_size
)
# disparity = stereo.compute(img1_rectified, img2_rectified)
disparity = stereo.compute(img1, img2)

# Display rectified images and disparity map
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(img1_rectified)
ax[1].imshow(img2_rectified)
ax[2].imshow((disparity - min_disparity) / num_disparities, cmap="gray")
plt.show()


# generate depth map with interpolated values
depth_map = np.zeros((img1.shape[0], img1.shape[1]))
for y in range(img1.shape[0]):
    for x in range(img1.shape[1]):
        # find the closest point in the original 2D point list: pts1
        closest_point_index = np.argmin(np.linalg.norm(np.array([x, y]) - pts1, axis=1))
        # get the depth value from the triangulated points
        depth_map[y, x] = triangulated_3d_points[closest_point_index, 2]

# normalize the depth map
depth_map = depth_map - np.min(depth_map)
depth_map = depth_map / np.max(depth_map)

plt.imshow(depth_map)
plt.show()
