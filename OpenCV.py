import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import argparse

def load_points(file3d, file2d):
    """
    Load corresponding 3D and 2D point sets from text files,
    skipping the first row (header) if present.
    """

    pts3d = np.loadtxt(file3d, skiprows=1)  # Skip header if present
    pts2d = np.loadtxt(file2d, skiprows=1)  # Skip header if present

    if pts3d.shape[0] != pts2d.shape[0]:
        raise ValueError("Number of points in 3D and 2D files do not match.")
    
    return pts3d, pts2d

def calibrate_openCV(file3d, file2d, image_size):
    """
    Perform camera calibration using OpenCV's calibrateCamera.
    - file3d: path to text file of 3D (X,Y,Z) coordinates
    - file2d: path to text file of 2D (x,y) pixel coordinates
    - image_size: tuple (width, height) of the image sensor/frame
    """
    # Load point correspondences
    pts3d, pts2d = load_points(file3d, file2d)
    # Reshape into the format OpenCV expects: list of Nx1x3 for 3D, Nx1x2 for 2D
    objp = pts3d.astype(np.float32).reshape(-1,1, 3)  
    imgp = pts2d.astype(np.float32).reshape(-1,1, 2)
    # 1. build a plausible starting intrinsic matrix assumes a pinhole model
    init_K = cv.initCameraMatrix2D([objp], [imgp], image_size)
    # 2. run calibration, telling it to use your guess
    #    - cameraMatrix=init_K tells OpenCV to refine this guess
    #    - distCoeffs=None initializes distortion to zero (will be estimated)
    #    - flags=cv.CALIB_USE_INTRINSIC_GUESS forces use of init_K as starting point
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    [objp], [imgp], image_size,
    cameraMatrix=init_K,
    distCoeffs=None,
    flags=cv.CALIB_USE_INTRINSIC_GUESS
    )
    # Convert first rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvecs[0])
    # Stack rotation and translation to form the 3×4 extrinsic matrix [R | t]
    Rt = np.hstack((R, tvecs[0]))    
    # compute full projection matrix        
    P = mtx.dot(Rt)
    print("OpenCV Calibration Matrix:\n", P)
    
    # Reprojection Error Calculation
    projected, _ = cv.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
    projected = projected.reshape(-1, 2)
    error = np.mean(np.linalg.norm(imgp.reshape(-1, 2) - projected, axis=1))
    print("Reprojection Error:", error)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration using OpenCV")
    # paths to input files
    parser.add_argument("file3d", type=str, help="Path to the 3D points file")
    parser.add_argument("file2d", type=str, help="Path to the 2D points file")
    # image size arguments optional
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    
    args = parser.parse_args()
    
    calibrate_openCV(args.file3d, args.file2d, (args.width, args.height))