# match.py
import sys
import time
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QMessageBox, QLineEdit, QFormLayout, QCheckBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Optional 3D viewer: Open3D; fallback to Matplotlib if not available
_HAS_O3D = True
try:
    import open3d as o3d  # pip install open3d
except Exception:
    _HAS_O3D = False
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt


def _bgr_to_qimage(img_bgr):
    h, w = img_bgr.shape[:2]
    return QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)


class StereoMatchApp(QMainWindow):
    def __init__(self, left_path=None, right_path=None):
        super().__init__()
        self.setWindowTitle('Stereo Dense Match + 3D Reconstruction')

        # ================== State ==================
        # Raw images
        self.left_img = None
        self.right_img = None
        self.left_gray = None
        self.right_gray = None

        # Chessboard / calibration state
        self.board_size = (9, 6)     # inner corners (cols, rows)
        self.square_size = 1.0       # units (e.g., mm if you know them)
        self.left_corners = None
        self.right_corners = None

        # Intrinsics / Distortion (defaults from user)
        self.K1 = np.array([[277.777777777778, 0.0, 100.0],
                            [0.0, 416.666666666667, 100.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self.K2 = self.K1.copy()
        self.D1 = np.zeros((5, 1), dtype=np.float64)
        self.D2 = np.zeros((5, 1), dtype=np.float64)

        # Extrinsics (relative pose)
        self.R = None
        self.T = None

        # Rectification / Projection / Reprojection
        self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None
        self.map1x = self.map1y = self.map2x = self.map2y = None
        self.left_rect = None
        self.right_rect = None
        self.left_rect_color = None

        # Dense disparity & 3D
        self.disp = None
        self.points3d = None
        self.points3d_mask = None
        self.colors3d = None

        # Options
        self.use_known_intrinsics = True    # we have K1,K2 from user
        self.use_known_pose = False         # set True to skip calibration and use baseline along x
        self.baseline_default = 0.25        # meters (used if use_known_pose=True and no R provided)

        # ================== UI ==================
        self._build_ui()
        self.showMaximized()

        if left_path and right_path:
            self._load_image(True, left_path)
            self._load_image(False, right_path)

    # -------------------- UI BUILD --------------------
    def _build_ui(self):
        # Image panes
        self.left_label = QLabel('Left Image');  self.left_label.setAlignment(Qt.AlignCenter)
        self.right_label = QLabel('Right Image'); self.right_label.setAlignment(Qt.AlignCenter)

        # File buttons
        btn_load_left  = QPushButton('Load Left')
        btn_load_left.clicked.connect(lambda: self._load_image(True))
        btn_load_right = QPushButton('Load Right')
        btn_load_right.clicked.connect(lambda: self._load_image(False))

        # Pattern controls
        self.board_edit = QLineEdit('9x6')
        self.square_edit = QLineEdit('1.0')

        # Intrinsics controls
        self.fx_edit = QLineEdit('277.777777777778')
        self.fy_edit = QLineEdit('416.666666666667')
        self.cx_edit = QLineEdit('100')
        self.cy_edit = QLineEdit('100')

        # Pose / baseline
        self.use_known_intrinsics_cb = QCheckBox('Use known intrinsics')
        self.use_known_intrinsics_cb.setChecked(self.use_known_intrinsics)
        self.use_known_intrinsics_cb.stateChanged.connect(self._on_known_intrinsics_toggle)

        self.use_known_pose_cb = QCheckBox('Use known pose (R=I, T=[B,0,0])')
        self.use_known_pose_cb.setChecked(self.use_known_pose)
        self.use_known_pose_cb.stateChanged.connect(self._on_known_pose_toggle)

        self.baseline_edit = QLineEdit(f'{self.baseline_default}')

        # Action buttons
        btn_detect     = QPushButton('1) Detect Chessboard')
        btn_detect.clicked.connect(self.detect_chessboard)

        btn_calib      = QPushButton('2) Calibrate (or Skip)')
        btn_calib.clicked.connect(self.calibrate_stereo)

        btn_compute_F  = QPushButton('Compute F (SIFT+RANSAC)')
        btn_compute_F.clicked.connect(self.compute_f_sift)

        btn_rectify    = QPushButton('3) Rectify')
        btn_rectify.clicked.connect(self.rectify_pair)

        btn_disparity  = QPushButton('4) Dense Disparity')
        btn_disparity.clicked.connect(self.compute_disparity)

        btn_recon      = QPushButton('5) Reconstruct 3D')
        btn_recon.clicked.connect(self.reconstruct_3d)

        btn_view3d     = QPushButton('View 3D')
        btn_view3d.clicked.connect(self.view_3d)

        btn_save_ply   = QPushButton('Save PLY')
        btn_save_ply.clicked.connect(self.save_ply)

        # Forms
        pattern_box = QGroupBox('Calibration pattern')
        pattern_form = QFormLayout()
        pattern_form.addRow('Board (CxR):', self.board_edit)
        pattern_form.addRow('Square size:', self.square_edit)
        pattern_box.setLayout(pattern_form)

        intr_box = QGroupBox('Intrinsics (Left/Right assumed same)')
        intr_grid = QGridLayout()
        intr_grid.addWidget(QLabel('fx'), 0, 0); intr_grid.addWidget(self.fx_edit, 0, 1)
        intr_grid.addWidget(QLabel('fy'), 0, 2); intr_grid.addWidget(self.fy_edit, 0, 3)
        intr_grid.addWidget(QLabel('cx'), 1, 0); intr_grid.addWidget(self.cx_edit, 1, 1)
        intr_grid.addWidget(QLabel('cy'), 1, 2); intr_grid.addWidget(self.cy_edit, 1, 3)
        intr_grid.addWidget(self.use_known_intrinsics_cb, 2, 0, 1, 4)
        intr_box.setLayout(intr_grid)

        pose_box = QGroupBox('Pose')
        pose_form = QFormLayout()
        pose_form.addRow(self.use_known_pose_cb)
        pose_form.addRow('Baseline B (m):', self.baseline_edit)
        pose_box.setLayout(pose_form)

        # Top buttons row
        top_btns = QHBoxLayout()
        for b in (btn_load_left, btn_load_right, btn_detect, btn_calib, btn_compute_F,
                  btn_rectify, btn_disparity, btn_recon, btn_view3d, btn_save_ply):
            top_btns.addWidget(b)

        # Images layout
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.left_label)
        img_layout.addWidget(self.right_label)

        # Left control column
        ctrl_col = QVBoxLayout()
        ctrl_col.addWidget(pattern_box)
        ctrl_col.addWidget(intr_box)
        ctrl_col.addWidget(pose_box)
        ctrl_col.addStretch(1)

        # Main layout (controls left, images right)
        main_layout = QHBoxLayout()
        ctrl_widget = QWidget(); ctrl_widget.setLayout(ctrl_col)
        imgs_widget = QWidget(); v = QVBoxLayout(); v.addLayout(top_btns); v.addLayout(img_layout); imgs_widget.setLayout(v)

        main_layout.addWidget(ctrl_widget, 0)
        main_layout.addWidget(imgs_widget, 1)

        container = QWidget(); container.setLayout(main_layout)
        self.setCentralWidget(container)

    # -------------------- Helpers --------------------
    def _on_known_intrinsics_toggle(self, state):
        self.use_known_intrinsics = (state == Qt.Checked)

    def _on_known_pose_toggle(self, state):
        self.use_known_pose = (state == Qt.Checked)

    def _update_board_params(self):
        try:
            txt = self.board_edit.text().lower().replace(' ', '')
            c, r = txt.split('x')
            self.board_size = (int(c), int(r))
        except Exception:
            QMessageBox.warning(self, 'Board Size', 'Invalid board format. Use like 9x6.')
            return False
        try:
            self.square_size = float(self.square_edit.text())
        except Exception:
            QMessageBox.warning(self, 'Square Size', 'Invalid square size.')
            return False
        return True

    def _update_intrinsics_from_ui(self):
        try:
            fx = float(self.fx_edit.text()); fy = float(self.fy_edit.text())
            cx = float(self.cx_edit.text()); cy = float(self.cy_edit.text())
        except Exception:
            QMessageBox.warning(self, 'Intrinsics', 'fx, fy, cx, cy must be numeric.')
            return False
        self.K1 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.K2 = self.K1.copy()
        # Keep zero distortion unless you change here:
        self.D1 = np.zeros((5, 1), dtype=np.float64)
        self.D2 = np.zeros((5, 1), dtype=np.float64)
        return True

    def _update_baseline_from_ui(self):
        try:
            self.baseline_default = float(self.baseline_edit.text())
        except Exception:
            QMessageBox.warning(self, 'Baseline', 'Baseline must be numeric.')
            return False
        return True

    def _show_img_on(self, img_bgr, lbl):
        qimg = _bgr_to_qimage(img_bgr)
        lbl.setPixmap(QPixmap.fromImage(qimg))
        lbl.setFixedSize(img_bgr.shape[1], img_bgr.shape[0])

    def _load_image(self, is_left, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open Image')
            if not path:
                return
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            QMessageBox.warning(self, 'Load Image', f'Failed to load: {path}')
            return
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if is_left:
            self.left_img = img_bgr
            self.left_gray = gray
            self._show_img_on(img_bgr, self.left_label)
        else:
            self.right_img = img_bgr
            self.right_gray = gray
            self._show_img_on(img_bgr, self.right_label)

    # -------------------- Chessboard detection --------------------
    def detect_chessboard(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Images', 'Load both images first.')
            return
        if not self._update_board_params():
            return

        t0 = time.time()
        pattern = self.board_size
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

        def find_corners(gray):
            if hasattr(cv2, 'findChessboardCornersSB'):
                ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
            else:
                ok, corners = cv2.findChessboardCorners(gray, pattern, None)
            if not ok:
                return False, None
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            corners = cv2.cornerSubPix(gray, np.float32(corners), (11, 11), (-1, -1), term)
            return True, corners

        okL, cornersL = find_corners(self.left_gray)
        okR, cornersR = find_corners(self.right_gray)
        if not okL or not okR:
            QMessageBox.warning(self, 'Chessboard', 'Could not find chessboard in both images.')
            return

        self.left_corners = cornersL
        self.right_corners = cornersR

        visL = self.left_img.copy()
        visR = self.right_img.copy()
        cv2.drawChessboardCorners(visL, pattern, cornersL, True)
        cv2.drawChessboardCorners(visR, pattern, cornersR, True)
        self._show_img_on(visL, self.left_label)
        self._show_img_on(visR, self.right_label)

        dt = time.time() - t0
        print(f'[Timing] Chessboard detection: {dt:.3f}s')
        QMessageBox.information(self, 'Chessboard', f'Detected {pattern[0]}x{pattern[1]} inner corners.\nTime: {dt:.3f}s')

    # -------------------- Stereo calibration --------------------
    def _object_points(self):
        cols, rows = self.board_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(self.square_size)
        return objp

    def calibrate_stereo(self):
        if self.left_corners is None or self.right_corners is None:
            if not self.use_known_pose:
                QMessageBox.warning(self, 'Calibration', 'Run "Detect Chessboard" first (or enable "Use known pose").')
                return

        if not self._update_board_params():
            return
        if not self._update_intrinsics_from_ui():
            return
        if not self._update_baseline_from_ui():
            return

        h, w = self.left_gray.shape if self.left_gray is not None else (None, None)

        # If user wants to fully skip calibration and use known pose (R=I, T=[B,0,0])
        if self.use_known_pose:
            self.R = np.eye(3, dtype=np.float64)
            self.T = np.array([self.baseline_default, 0.0, 0.0], dtype=np.float64)
            print('Using provided K, zero distortion, R=I, T=[B,0,0]. Skipping stereoCalibrate.')
            QMessageBox.information(self, 'Calibration', 'Skipped — using known pose (R=I, T=[B,0,0]).')
            return

        # Else we estimate only R,T (since we know K and keep D fixed at zeros)
        t0 = time.time()
        objp = self._object_points()
        objpoints = [objp]
        imgpoints1 = [self.left_corners.reshape(-1, 1, 2)]
        imgpoints2 = [self.right_corners.reshape(-1, 1, 2)]

        flags = (cv2.CALIB_FIX_INTRINSIC |
                 cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            self.K1, self.D1, self.K2, self.D2, (w, h),
            flags=flags, criteria=criteria
        )

        self.K1, self.D1, self.K2, self.D2 = K1, D1, K2, D2
        self.R, self.T, self.E, self.F = R, T, E, F

        dt = time.time() - t0
        baseline = float(np.linalg.norm(T))
        fx = float(K1[0, 0]); cx, cy = float(K1[0, 2]), float(K1[1, 2])

        msg = (f"Stereo RMS reprojection error: {rms:.6f}\n"
               f"Baseline (||T||): {baseline:.6f} (same units as square size)\n\n"
               f"K1:\n{K1}\nD1:\n{D1.ravel()}\n\n"
               f"K2:\n{K2}\nD2:\n{D2.ravel()}\n\n"
               f"R:\n{R}\nT:\n{T.ravel()}\n\n"
               f"F:\n{F}\nE:\n{E}\n\n"
               f"(fx, cx, cy) from left: ({fx:.3f}, {cx:.3f}, {cy:.3f})\n"
               f"Time: {dt:.3f}s")
        print(msg)
        QMessageBox.information(self, 'Calibration Results', msg)

    # -------------------- Fundamental matrix by SIFT (debug) --------------------
    def compute_f_sift(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Load both images first')
            return
        t0 = time.time()
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.left_gray, None)
        kp2, des2 = sift.detectAndCompute(self.right_gray, None)
        if des1 is None or des2 is None:
            QMessageBox.warning(self, 'F', 'SIFT could not find descriptors.')
            return
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            QMessageBox.warning(self, 'F', 'Not enough matches for F.')
            return
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        if F is None:
            QMessageBox.warning(self, 'F', 'Fundamental matrix estimation failed.')
            return
        inliers = int(mask.sum())
        dt = time.time() - t0
        print(f"F via SIFT+RANSAC inliers: {inliers}/{len(mask)}\nF =\n{F}\nTime: {dt:.3f}s")
        QMessageBox.information(self, 'Fundamental Matrix', f'F =\n{F}\nInliers: {inliers}/{len(mask)}\nTime: {dt:.3f}s')
        self.F = F

    # -------------------- Rectification --------------------
    def rectify_pair(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Rectify', 'Load both images first.')
            return
        if not self._update_intrinsics_from_ui():
            return
        if self.use_known_pose and (self.R is None or self.T is None):
            # set default R,T if user toggled known pose but hasn't calibrated
            if not self._update_baseline_from_ui():
                return
            self.R = np.eye(3, dtype=np.float64)
            self.T = np.array([self.baseline_default, 0.0, 0.0], dtype=np.float64)

        if any(x is None for x in [self.K1, self.D1, self.K2, self.D2]):
            QMessageBox.warning(self, 'Rectify', 'Intrinsics missing.')
            return
        if any(x is None for x in [self.R, self.T]):
            QMessageBox.warning(self, 'Rectify', 'Pose missing. Calibrate or enable "Use known pose".')
            return

        t0 = time.time()
        h, w = self.left_gray.shape
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, (w, h), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, (w, h), cv2.CV_32FC1)

        self.left_rect = cv2.remap(self.left_gray, self.map1x, self.map1y, cv2.INTER_LINEAR)
        self.right_rect = cv2.remap(self.right_gray, self.map2x, self.map2y, cv2.INTER_LINEAR)
        self.left_rect_color = cv2.remap(self.left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)

        left_vis = cv2.cvtColor(self.left_rect, cv2.COLOR_GRAY2BGR)
        right_vis = cv2.cvtColor(self.right_rect, cv2.COLOR_GRAY2BGR)
        for y in range(0, left_vis.shape[0], max(1, left_vis.shape[0] // 10)):
            cv2.line(left_vis,  (0, y), (left_vis.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.line(right_vis, (0, y), (right_vis.shape[1] - 1, y), (0, 255, 0), 1)
        self._show_img_on(left_vis, self.left_label)
        self._show_img_on(right_vis, self.right_label)

        dt = time.time() - t0
        print(f'[Timing] Rectification: {dt:.3f}s')
        QMessageBox.information(self, 'Rectify', f'Rectification complete. Q available.\nTime: {dt:.3f}s')

    # -------------------- Dense disparity --------------------
    def compute_disparity(self):
        if self.left_rect is None or self.right_rect is None:
            QMessageBox.warning(self, 'Disparity', 'Rectify first.')
            return

        t0 = time.time()
        min_disp = 0
        num_disp = 16 * 12  # must be divisible by 16
        block = 5
        sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 3 * block * block,
            P2=32 * 3 * block * block,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32,
            disp12MaxDiff=1
        )
        disp = sgbm.compute(self.left_rect, self.right_rect).astype(np.float32) / 16.0
        self.disp = disp

        disp_vis = (disp - min_disp) / float(num_disp)
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        cv2.imshow('Disparity', disp_vis)
        cv2.waitKey(1)

        dt = time.time() - t0
        print(f'[Timing] Disparity: {dt:.3f}s')
        QMessageBox.information(self, 'Disparity', f'Dense disparity computed.\nTime: {dt:.3f}s')

    # -------------------- 3D reconstruction --------------------
    def reconstruct_3d(self):
        if self.disp is None or self.Q is None:
            QMessageBox.warning(self, '3D', 'Compute disparity and ensure Q is available (Rectify first).')
            return

        t0 = time.time()
        points3d = cv2.reprojectImageTo3D(self.disp, self.Q)  # HxWx3
        mask = self.disp > 0.0  # valid disparities
        self.points3d = points3d
        self.points3d_mask = mask

        if self.left_rect_color is None:
            self.left_rect_color = cv2.cvtColor(self.left_rect, cv2.COLOR_GRAY2BGR)
        colors = cv2.cvtColor(self.left_rect_color, cv2.COLOR_BGR2RGB)
        self.colors3d = colors

        valid = int(mask.sum())
        total = mask.size
        dt = time.time() - t0
        print(f'[Timing] Reproject 3D: {dt:.3f}s')
        print(f"3D points reconstructed for {valid}/{total} pixels (valid disparity).")
        QMessageBox.information(self, '3D Reconstruction',
                                f'Reconstructed {valid} 3D points (of {total}).\nTime: {dt:.3f}s')

    # -------------------- View / Save point cloud --------------------
    def _gather_cloud(self):
        if self.points3d is None or self.points3d_mask is None:
            return None, None
        pts = self.points3d[self.points3d_mask]
        cols = self.colors3d[self.points3d_mask]
        # Filter NaNs/Infs
        good = np.isfinite(pts).all(axis=1)
        pts = pts[good]
        cols = cols[good]
        cols = (cols.astype(np.float32) / 255.0)
        return pts, cols

    def view_3d(self):
        pts, cols = self._gather_cloud()
        if pts is None or pts.shape[0] == 0:
            QMessageBox.warning(self, '3D Viewer', 'No valid 3D points to display.')
            return

        if _HAS_O3D:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(cols)
            o3d.visualization.draw_geometries([pc], window_name='3D Reconstruction')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            s = max(1, 100000 // max(1, pts.shape[0]))
            ax.scatter(pts[::s, 0], pts[::s, 1], pts[::s, 2], s=1)
            ax.set_title('3D Reconstruction (Matplotlib fallback)')
            plt.show()

    def save_ply(self):
        pts, cols = self._gather_cloud()
        if pts is None or pts.shape[0] == 0:
            QMessageBox.warning(self, 'Save PLY', 'No valid 3D points to save.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Point Cloud', filter='PLY (*.ply)')
        if not path:
            return
        if _HAS_O3D:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(cols)
            o3d.io.write_point_cloud(path, pc, write_ascii=True)
        else:
            with open(path, 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write(f'element vertex {pts.shape[0]}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
                f.write('end_header\n')
                for p, c in zip(pts, (cols * 255).astype(np.uint8)):
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        QMessageBox.information(self, 'Save PLY', f'Saved: {path}')


if __name__ == '__main__':
    left = sys.argv[1] if len(sys.argv) > 1 else None
    right = sys.argv[2] if len(sys.argv) > 2 else None
    app = QApplication(sys.argv)
    win = StereoMatchApp(left, right)
    win.show()
    sys.exit(app.exec_())