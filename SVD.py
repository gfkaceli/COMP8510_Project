import numpy as np
import argparse
import matplotlib.pyplot as plt


def load_points(file2d, file3d):
    """
    Load corresponding 3D and 2D point sets from text files,
    skipping the first row (header) if present.
    """
    pts3d = np.loadtxt(file3d, skiprows=1)  # Skip header if present
    pts2d = np.loadtxt(file2d, skiprows=1)  # Skip header if present

    if pts3d.shape[0] != pts2d.shape[0]:
        raise ValueError("Number of points in 3D and 2D files do not match.")
    
    return pts3d, pts2d

def calibrate_svd(file2d, file3d):
    """
    Perform Direct Linear Transform (DLT) via SVD to find the 3×4 projection matrix.
    - file3d: path to text file of 3D (X,Y,Z) coordinates
    - file2d: path to text file of 2D (x,y) pixel coordinates
    """

    # load the points
    pts3d, pts2d = load_points(file3d, file2d)

    N = pts3d.shape[0] # number of correspondences
    # Build the linear system A * p = 0, where p is the 12-vector of projection matrix entries
    A = []

    # For each correspondence we add two rows to A:
    #  [ X  Y  Z  1   0  0  0  0  -xX  -xY  -xZ  -x ]
    #  [ 0  0  0  0   X  Y  Z  1  -yX  -yY  -yZ  -y ]
    for (X, Y, Z), (x, y) in zip(pts3d, pts2d):
        A.append([X, Y, Z, 1, 0,0,0,0, -x*X, -x*Y, -x*Z, -x])
        A.append([0,0,0,0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])

    A = np.array(A)
    # Perform SVD on A: A = U * S * Vt
    # The solution p (flattened projection matrix) is the last row of Vt (smallest singular value)

    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3,4)
    P = P / P[-1,-1]
    print("SVD Calibration Matrix (3x4):")
    print(P)
    # Compute reprojection error
    pts3d_h = np.hstack((pts3d, np.ones((N,1))))
    proj = (P @ pts3d_h.T).T
    proj = proj[:,:2] / proj[:,2,np.newaxis]
    error = np.linalg.norm(proj - pts2d, axis=1)
    print(f"Average Reprojection Error: {np.mean(error):.9f} pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera using 2D-3D point correspondences.")
    parser.add_argument("file2d", type=str, help="File containing 2D points (x, y).")
    parser.add_argument("file3d", type=str, help="File containing 3D points (X, Y, Z).")
    args = parser.parse_args()
    
    calibrate_svd(args.file2d, args.file3d)