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

# GUI app for hybrid dense stereo matching
class StereoMatchApp(QMainWindow):
    def __init__(self, left_path=None, right_path=None):
        super().__init__()
        self.setWindowTitle('Stereo Dense Match GUI')

        # Image data
        self.left_img = None
        self.right_img = None
        self.left_gray = None
        self.right_gray = None

        # Match data
        self.edge_matches = {}
        self.dense_matches = {}
        self.total_matches = 0

        # UI setup
        self.left_label = QLabel('Left Image')
        self.left_label.setAlignment(Qt.AlignCenter)
        self.right_label = QLabel('Right Image')
        self.right_label.setAlignment(Qt.AlignCenter)

        btn_load_left  = QPushButton('Load Left')
        btn_load_left.clicked.connect(lambda: self.load_image(True))
        btn_load_right = QPushButton('Load Right')
        btn_load_right.clicked.connect(lambda: self.load_image(False))
        btn_compute_F  = QPushButton('Compute F')
        btn_compute_F.clicked.connect(self.compute_f)
        btn_match      = QPushButton('Run Matching')
        btn_match.clicked.connect(self.run_matching)

        btn_layout = QHBoxLayout()
        for btn in (btn_load_left, btn_load_right, btn_compute_F, btn_match):
            btn_layout.addWidget(btn)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.left_label)
        img_layout.addWidget(self.right_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.showMaximized()

        if left_path and right_path:
            self.load_image(True, left_path)
            self.load_image(False, right_path)

    def load_image(self, is_left, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open Image')
            if not path:
                return
        pil = PILImage.open(path).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display
        h, w = gray.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        lbl = self.left_label if is_left else self.right_label
        lbl.setPixmap(pix)
        lbl.setFixedSize(w, h)

        # Store
        if is_left:
            self.left_img = img
            self.left_gray = gray
        else:
            self.right_img = img
            self.right_gray = gray

    def compute_f(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Load both images first')
            return
        # SIFT + BF + RANSAC
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.left_gray, None)
        kp2, des2 = sift.detectAndCompute(self.right_gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None:
            QMessageBox.warning(self, 'Error', 'Fundamental matrix failed')
            return
        # Keep inliers only
        self.pts1 = pts1[mask.ravel()==1]
        self.pts2 = pts2[mask.ravel()==1]
        self.F = F
        # Display actual F matrix
        QMessageBox.information(self, 'Fundamental Matrix', f'F =\n{self.F}')

    def run_matching(self):
        if not hasattr(self, 'F'):
            QMessageBox.warning(self, 'Error', 'Compute F first')
            return
        # Rectify
        h, w = self.left_gray.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(self.pts1, self.pts2, self.F, (w, h))
        left_rect  = cv2.warpPerspective(self.left_gray, H1, (w, h))
        right_rect = cv2.warpPerspective(self.right_gray, H2, (w, h))
        left_color = cv2.warpPerspective(self.left_img, H1, (w, h))

        # Edge mask via Sobel
        gx = cv2.Sobel(left_rect, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(left_rect, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        edge_mask = (mag > 50).astype(np.uint8)

        # ZNCC on edge pixels
        patch, drange = 11, 100
        win = patch//2
        self.edge_matches = {}
        for y in range(win, h-win):
            xs = np.union1d(np.where(edge_mask[y]==1)[0], [win, w-win-1])
            for x in xs:
                if x-win<0 or x+win>=w: continue
                tpl = left_rect[y-win:y+win+1, x-win:x+win+1]
                m1, s1 = tpl.mean(), tpl.std()
                if s1==0: continue
                best, x2b = -1, None
                lo = max(win, x-drange); hi = min(w-win-1, x+drange)
                for x2 in range(lo, hi+1):
                    pr = right_rect[y-win:y+win+1, x2-win:x2+win+1]
                    m2, s2 = pr.mean(), pr.std()
                    if s2==0: continue
                    zncc = ((tpl-m1)*(pr-m2)).sum()/(s1*s2*patch*patch)
                    if zncc>best:
                        best, x2b = zncc, x2
                if x2b is not None:
                    self.edge_matches[(x,y)] = (x2b,y)

        # Line-segment interpolation
        self.dense_matches = {}
        seg = {}
        for (x,y),(x2,y2) in self.edge_matches.items():
            seg.setdefault(y, {})[x] = x2
        for y, row in seg.items():
            xs = sorted(row.keys())
            for i in range(len(xs)-1):
                xA, xB = xs[i], xs[i+1]
                xA2, xB2 = row[xA], row[xB]
                L = xB-xA
                if L<=1: continue
                lam = (xB2 - xA2)/L
                for x in range(xA+1, xB):
                    x2 = int(xA2 + lam*(x-xA))
                    self.dense_matches[(x,y)] = (x2,y)

        # Print stats in terminal
        ecount = len(self.edge_matches)
        dcount = len(self.dense_matches)
        self.total_matches = ecount + dcount
        print(f"Edges: {ecount}\nDense: {dcount}\nTotal: {self.total_matches}")

        # Display disparity & reconstruction
        disp = np.zeros((h,w), np.float32)
        for (x,y),(x2,y2) in self.edge_matches.items(): disp[y,x] = x-x2
        for (x,y),(x2,y2) in self.dense_matches.items(): disp[y,x] = x-x2
        disp_vis = cv2.normalize(disp, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
        map_y, map_x = np.indices((h,w), dtype=np.float32)
        recon = cv2.remap(left_color,
                          (map_x - disp).astype(np.float32),
                          map_y,
                          cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cv2.imshow('Disparity', disp_vis)
        cv2.imshow('Reconstruction', recon)
        cv2.waitKey(1)

if __name__=='__main__':
    left = sys.argv[1] if len(sys.argv)>1 else None
    right= sys.argv[2] if len(sys.argv)>2 else None
    app = QApplication(sys.argv)
    win = StereoMatchApp(left, right)
    win.show()
    sys.exit(app.exec_())
