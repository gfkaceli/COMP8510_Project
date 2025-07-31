import sys
import cv2
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Maximum display size for loaded images
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600

class ClickableLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.callback = None

    def mousePressEvent(self, event):
        if self.callback:
            self.callback(event)

class StereoEpipolarApp(QMainWindow):
    def __init__(self, left_path=None, right_path=None):
        super().__init__()
        self.setWindowTitle('Stereo Epipolar Geometry & Dense Matching')

        # Images & data
        self.left_img = None
        self.right_img = None
        self.left_gray = None
        self.right_gray = None
        self.F = None
        self.pts1 = None
        self.pts2 = None

        # Parameters
        self.patch_size = 11          # window size for ZNCC
        self.disparity_range = 100    # max horizontal search range
        self.edge_thresh = 50         # Sobel threshold for edges

        # UI setup
        self.left_label = ClickableLabel()
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.callback = self.on_left_click
        self.right_label = QLabel()
        self.right_label.setAlignment(Qt.AlignCenter)

        # Buttons
        btn_load_left  = QPushButton('Load Left Image')
        btn_load_left.clicked.connect(lambda: self.load_image(is_left=True))
        btn_load_right = QPushButton('Load Right Image')
        btn_load_right.clicked.connect(lambda: self.load_image(is_left=False))
        btn_compute_F  = QPushButton('Compute Fundamental Matrix')
        btn_compute_F.clicked.connect(self.compute_fundamental)
        btn_dense      = QPushButton('Dense Match & Reconstruct')
        btn_dense.clicked.connect(self.compute_dense)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_load_left)
        btn_layout.addWidget(btn_load_right)
        btn_layout.addWidget(btn_compute_F)
        btn_layout.addWidget(btn_dense)

        # Layout assembly
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_label)
        image_layout.addWidget(self.right_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.showMaximized()

        # Optionally load via CLI args
        if left_path and right_path:
            self.load_image(is_left=True,  path=left_path)
            self.load_image(is_left=False, path=right_path)

    def load_image(self, is_left, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open Image')
            if not path:
                return
        pil = PILImage.open(path)
        pil.info.pop('icc_profile', None)
        pil = pil.convert('RGB')
        rgb = np.array(pil)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # downscale if needed
        scale = min(MAX_DISPLAY_WIDTH/w, MAX_DISPLAY_HEIGHT/h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w*scale), int(h*scale)
            img  = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.resize(gray,(new_w,new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        qimg   = QImage(img.data, w, h, 3*w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)

        if is_left:
            self.left_img  = img
            self.left_gray = gray
            self.left_label.setPixmap(pixmap)
            self.left_label.setFixedSize(w, h)
        else:
            self.right_img  = img
            self.right_gray = gray
            self.right_label.setPixmap(pixmap)
            self.right_label.setFixedSize(w, h)

    def compute_fundamental(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Please load both images first.')
            return
        # Feature detection & matching
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.left_gray,  None)
        kp2, des2 = sift.detectAndCompute(self.right_gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        self.pts1, self.pts2 = pts1, pts2
        self.F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if self.F is not None:
            QMessageBox.information(self, 'Fundamental Matrix', f'F =\n{self.F}')

    def compute_dense(self):
        if self.F is None or self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Compute fundamental matrix first.')
            return
        h, w = self.left_gray.shape
        # Rectify images
        _, H1, H2 = cv2.stereoRectifyUncalibrated(self.pts1, self.pts2, self.F, (w, h))
        left_rect  = cv2.warpPerspective(self.left_gray,  H1, (w, h))
        right_rect = cv2.warpPerspective(self.right_gray, H2, (w, h))
        left_disp_color = cv2.warpPerspective(self.left_img, H1, (w, h))

        # 1. Edge segmentation
        sobelx    = cv2.Sobel(left_rect, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(left_rect, cv2.CV_64F, 0, 1, ksize=3)
        mag       = np.hypot(sobelx, sobely)
        edge_mask = cv2.Canny(left_rect, 20,100) #(mag > self.edge_thresh).astype(np.uint8)

        # Prepare disparity map
        disp           = np.zeros((h, w), dtype=np.float32)
        win            = self.patch_size // 2
        matches_map    = {}    # for mapping segments
        edge_matches   = []    # list of explicit edge matches
        dense_matches  = []    # list of interpolated matches

        # 2. Classical ZNCC matching on edge pixels & row boundaries
        for y in range(win, h-win):
            edge_x    = np.where(edge_mask[y] == 1)[0]
            boundary_x= np.array([win, w-win-1])
            xs = np.union1d(edge_x, boundary_x)
            for x in xs:
                if x-win < 0 or x+win >= w:
                    continue
                tpl = left_rect[y-win:y+win+1, x-win:x+win+1]
                t_mean, t_std = tpl.mean(), tpl.std()
                if t_std == 0 or np.isnan(t_std):
                    continue
                best_score, best_x2 = -1.0, None
                x_lo = max(win, x - self.disparity_range)
                x_hi = min(w-win-1, x + self.disparity_range)
                for x2 in range(x_lo, x_hi+1):
                    patch = right_rect[y-win:y+win+1, x2-win:x2+win+1]
                    p_mean, p_std = patch.mean(), patch.std()
                    if p_std == 0 or np.isnan(p_std):
                        continue
                    zncc = ((tpl - t_mean)*(patch - p_mean)).sum()/(t_std*p_std*self.patch_size**2)
                    if zncc > best_score:
                        best_score, best_x2 = zncc, x2
                if best_x2 is not None:
                    disp[y, x] = x - best_x2
                    matches_map.setdefault(y, {})[x] = best_x2
                    edge_matches.append(((x, y), (best_x2, y)))

        # 3. Segment-based interpolation for non-edge regions
        for y, row_matches in matches_map.items():
            xs = np.array(sorted(row_matches.keys()))
            for i in range(len(xs) - 1):
                xA, xB   = xs[i], xs[i+1]
                xA2, xB2 = row_matches[xA], row_matches[xB]
                length   = xB - xA
                if length <= 1:
                    continue
                scale = (xB2 - xA2) / float(length)
                for x in range(xA+1, xB):
                    x2_interp = xA2 + scale * (x - xA)
                    disp[y, x] = x - x2_interp
                    dense_matches.append(((x, y), (int(x2_interp), y)))

        # 4. Output match lists
        print("Edge Matches (left->right):", len(edge_matches))
        print("Dense Matches (left->right):", len(dense_matches))

        # Visualize disparity
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow('Disparity', disp_vis.astype(np.uint8))

        # Reconstruct right image by remapping left_color through disparity
        h_map, w_map = np.indices((h, w), dtype=np.float32)
        map_x = (w_map - disp).astype(np.float32)
        map_y = h_map.astype(np.float32)
        recon = cv2.remap(
            left_disp_color, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imshow('Reconstructed Right', recon)
        cv2.waitKey(1)

    def on_left_click(self, event):
        # interactive epipolar-line view (unused here)
        pass

if __name__ == '__main__':
    left = sys.argv[1] if len(sys.argv) > 1 else None
    right = sys.argv[2] if len(sys.argv) > 2 else None
    app = QApplication(sys.argv)
    window = StereoEpipolarApp(left, right)
    window.show()
    sys.exit(app.exec_())

