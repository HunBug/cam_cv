from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class CameraCalibParams:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray


class ICamCalib(ABC):
    def __init__(self, checkerboard_pattern: Tuple[int, int], square_size: float = 1.0):
        self.checkerboard_pattern = checkerboard_pattern
        self.square_size = square_size


    def get_3d_points(self) -> np.ndarray:
        points = np.zeros((self.checkerboard_pattern[0] * self.checkerboard_pattern[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:self.checkerboard_pattern[0], 0:self.checkerboard_pattern[1]].T.reshape(-1, 2)
        return points * self.square_size


    def calibrate_camera(self, checkerboard_image: np.ndarray) -> CameraCalibParams:
        corners = self.find_chessboard_corners(checkerboard_image)
        if corners is None:
            raise ValueError("Could not find chessboard corners in one of the images")
        points_3d = self.get_3d_points()
        return self.calibrate_camera_with_points(points_3d, corners, checkerboard_image.shape[:2])
    
    def calibrate_camera_multiple_image(self, checkerboard_images: List[np.ndarray]) -> CameraCalibParams:
        points_3ds = []
        points_2ds = []
        for image in checkerboard_images:
            corners = self.find_chessboard_corners(image)
            if corners is None:
                raise ValueError("Could not find chessboard corners in one of the images")
            points_3ds.append(self.get_3d_points())
            points_2ds.append(corners)
        return self.calibrate_camera_with_points_multiple(points_3ds, points_2ds, checkerboard_images[0].shape[:2])
            

    @abstractmethod
    def find_chessboard_corners(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def undistort_image(image: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def undistort_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def project_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        pass

    @staticmethod
    def get_projection_error(points_3d_space: np.ndarray, points_2d_image: np.ndarray, camera_calib_params: CameraCalibParams) -> float:
        points_2d_projected = ICamCalib.project_points(points_3d_space, camera_calib_params)
        return np.linalg.norm(points_2d_image - points_2d_projected)
    
    @abstractmethod
    def calibrate_camera_with_points(self, points_3d: np.ndarray, corners: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        pass


class OpenCvCalibration(ICamCalib):
    def __init__(self, checkerboard_pattern: Tuple[int, int], square_size: float = 1.0):
        super().__init__(checkerboard_pattern, square_size)

    def calibrate_camera_with_points(self, points_3d: np.ndarray, corners: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([points_3d], [corners], image_shape, None, None)
        return CameraCalibParams(camera_matrix, dist_coeffs, rvecs[0], tvecs[0])

    @staticmethod
    def find_chessboard_corners(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, (9, 6), None)[1]

    @staticmethod
    def undistort_image(image: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return cv2.undistort(image, camera_calib_params.camera_matrix, camera_calib_params.dist_coeffs)

    @staticmethod
    def undistort_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return cv2.undistortPoints(points, camera_calib_params.camera_matrix, camera_calib_params.dist_coeffs)

    @staticmethod
    def project_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        points, _ = cv2.projectPoints(points, camera_calib_params.rvecs, camera_calib_params.tvecs, camera_calib_params.camera_matrix, camera_calib_params.dist_coeffs)
        return points
    

class MinimalCameraCalibration(ICamCalib):
    def __init__(self, checkerboard_pattern: Tuple[int, int], square_size: float = 1.0):
        super().__init__(checkerboard_pattern, square_size)

    def calibrate_camera_with_points(self, points_3d: np.ndarray, corners: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        points_2d = corners.reshape(-1, 2)
        return self.calibrate_camera_with_points_2d(points_3d, points_2d, image_shape)
    
    def calibrate_camera_with_points_2d(self, points_3d: np.ndarray, points_2d: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        points_3d = self.get_3d_points()
        camera_matrix = self.get_camera_matrix(points_3d, points_2d, image_shape)
        dist_coeffs = self.get_dist_coeffs(points_3d, points_2d, camera_matrix)
        rvecs, tvecs = self.get_rvecs_tvecs(points_3d, points_2d, camera_matrix, dist_coeffs)
        return CameraCalibParams(camera_matrix, dist_coeffs, rvecs, tvecs)
    
    @staticmethod
    def get_camera_matrix(points_3d: np.ndarray, points_2d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        return np.array([[image_shape[1], 0, image_shape[1] / 2], [0, image_shape[1], image_shape[0] / 2], [0, 0, 1]])
    
    @staticmethod
    def get_dist_coeffs(points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        return np.zeros((1, 5))
    
    @staticmethod
    def get_rvecs_tvecs(points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, rvecs, tvecs = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
        return rvecs, tvecs
    
    @staticmethod
    def find_chessboard_corners(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, (9, 6), None)[1]
    
    @staticmethod
    def undistort_image(image: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return image
    
    @staticmethod
    def undistort_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return points
    
    @staticmethod
    def project_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        points, _ = cv2.projectPoints(points, camera_calib_params.rvecs, camera_calib_params.tvecs, camera_calib_params.camera_matrix, camera_calib_params.dist_coeffs)
        return points
    

# Work only with non-coplanar points
class BasicCameraCalibration(ICamCalib):
    def __init__(self, checkerboard_pattern: Tuple[int, int], square_size: float = 1.0):
        super().__init__(checkerboard_pattern, square_size)

    def calibrate_camera_with_points(self, points_3d: np.ndarray, corners: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        points_2d = corners.reshape(-1, 2)
        return self.calibrate_camera_with_points_2d(points_3d, points_2d, image_shape)
    
    def calibrate_camera_with_points_2d(self, points_3d: np.ndarray, points_2d: np.ndarray, image_shape: Tuple[int, int]) -> CameraCalibParams:
        points_3d = self.get_3d_points()
        camera_matrix, rvecs, tvecs = self.solve_camera_matrix(points_3d, points_2d)
        dist_coeffs = self.get_dist_coeffs(points_3d, points_2d, camera_matrix)
        return CameraCalibParams(camera_matrix, dist_coeffs, rvecs, tvecs)

    def solve_camera_matrix(self, points_3d, points_2d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # solve for camera matrix
        A = np.zeros((2 * points_3d.shape[0], 12))
        for i in range(points_3d.shape[0]):
            A[2*i, 0:3] = points_3d[i]
            A[2*i, 3] = 1 # switch to homogeneous coordinates
            A[2*i, 8:11] = -points_2d[i, 0] * points_3d[i]
            A[2*i, 11] = -points_2d[i, 0]
            A[2*i+1, 4:7] = points_3d[i]
            A[2*i+1, 7] = 1 # switch to homogeneous coordinates
            A[2*i+1, 8:11] = -points_2d[i, 1] * points_3d[i]
            A[2*i+1, 11] = -points_2d[i, 1]

        _, _, V = np.linalg.svd(A)
        camera_matrix = V[-1, :].reshape(3, 4)
        camera_matrix = camera_matrix / camera_matrix[-1, -1]

        print(camera_matrix)
        # decompose camera matrix with QR decomposition
        K, R = np.linalg.qr(camera_matrix[:, 0:3])
        t = np.linalg.inv(K) @ camera_matrix[:, 3]

        K = K / K[2, 2]

        return K, R, t

    @staticmethod
    def get_dist_coeffs(points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        return np.zeros((1, 5))
       
    @staticmethod
    def find_chessboard_corners(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, (9, 6), None)[1]
    
    @staticmethod
    def undistort_image(image: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return image
    
    @staticmethod
    def undistort_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        return points
    
    @staticmethod
    def project_points(points: np.ndarray, camera_calib_params: CameraCalibParams) -> np.ndarray:
        points, _ = cv2.projectPoints(points, camera_calib_params.rvecs, camera_calib_params.tvecs, camera_calib_params.camera_matrix, camera_calib_params.dist_coeffs)
        return points