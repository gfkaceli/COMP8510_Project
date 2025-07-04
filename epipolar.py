import sys
import cv2
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
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
        self.setWindowTitle('Stereo Epipolar Geometry')

        # Image placeholders
        self.left_img = None
        self.right_img = None
        self.left_gray = None
        self.right_gray = None
        self.F = None
        self.patch_size = 11  # patch size for ZNCC
        self.disparity_range = 100  # max disparity search range
        self.patch_display_scale = 10  # how much to enlarge patches for display

        # UI setup
        self.left_label = ClickableLabel()
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.callback = self.on_left_click
        self.right_label = QLabel()
        self.right_label.setAlignment(Qt.AlignCenter)

        btn_load_left = QPushButton('Load Left Image')
        btn_load_left.clicked.connect(lambda: self.load_image(is_left=True))
        btn_load_right = QPushButton('Load Right Image')
        btn_load_right.clicked.connect(lambda: self.load_image(is_left=False))
        btn_compute_F = QPushButton('Compute Fundamental Matrix')
        btn_compute_F.clicked.connect(self.compute_fundamental)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.left_label)
        hlayout.addWidget(self.right_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_load_left)
        btn_layout.addWidget(btn_load_right)
        btn_layout.addWidget(btn_compute_F)

        # Patch display labels
        self.template_view = QLabel('Template')
        self.match_view = QLabel('Match')
        view_w = self.patch_size * self.patch_display_scale
        self.template_view.setFixedSize(view_w, view_w)
        self.match_view.setFixedSize(view_w, view_w)
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(self.template_view)
        patch_layout.addWidget(self.match_view)

        main_layout = QVBoxLayout()
        main_layout.addLayout(hlayout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(patch_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.showMaximized()

        # If paths provided via command line, load them
        if left_path and right_path:
            self.load_image(is_left=True, path=left_path)
            self.load_image(is_left=False, path=right_path)

    def load_image(self, is_left, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open Image')
            if not path:
                return
        # Load with PIL to strip incorrect ICC profiles
        pil = PILImage.open(path)
        pil.info.pop('icc_profile', None)
        pil = pil.convert('RGB')
        rgb = np.array(pil)
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Grayscale version
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Resize if larger than max display size
        scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Convert to QPixmap for display
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)

        if is_left:
            self.left_img = img
            self.left_gray = gray
            self.left_label.setFixedSize(w, h)
            self.left_label.setPixmap(pixmap)
        else:
            self.right_img = img
            self.right_gray = gray
            self.right_pixmap = pixmap
            self.right_label.setFixedSize(w, h)
            self.right_label.setPixmap(pixmap)

    def compute_fundamental(self):
        if self.left_gray is None or self.right_gray is None:
            return
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.left_gray, None)
        kp2, des2 = sift.detectAndCompute(self.right_gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        self.F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if self.F is not None:
            print("Fundamental matrix F:\n", self.F)
            QMessageBox.information(self, "Fundamental Matrix", f"F =\n{self.F}")
            self.statusBar().showMessage('Fundamental matrix computed')

    def on_left_click(self, event):
        if self.F is None:
            return
        x, y = event.pos().x(), event.pos().y()
        self.find_corresponding(x, y)

    def find_corresponding(self, x, y):
        p = np.array([x, y, 1.0])
        l = self.F.dot(p)
        a, b, c = l
        h, w = self.right_gray.shape
        win = self.patch_size // 2
        if x-win<0 or x+win>=w or y-win<0 or y+win>=h:
            return
        template = self.left_gray[y-win:y+win+1, x-win:x+win+1]
        t_mean, t_std = np.mean(template), np.std(template)
        best_score, best_pt, best_patch = -1.0, None, None
        for x2 in range(win, w-win):
            if abs(x2 - x) > self.disparity_range: continue
            y2 = int(round(-(a*x2 + c) / b)) if b != 0 else 0
            if y2 < win or y2 >= h-win: continue
            patch = self.right_gray[y2-win:y2+win+1, x2-win:x2+win+1]
            p_mean, p_std = np.mean(patch), np.std(patch)
            if t_std == 0 or p_std == 0: continue
            zncc = np.sum((template-t_mean)*(patch-p_mean)) / (t_std*p_std*self.patch_size**2)
            if zncc > best_score:
                best_score, best_pt, best_patch = zncc, (x2, y2), patch.copy()
        if best_pt:
            self.draw_epipolar_and_marker(a, b, c, best_pt)
            self.update_patch_views(template, best_patch)
            self.statusBar().showMessage(f"Match at {best_pt}, ZNCC={best_score:.3f}")

    def update_patch_views(self, template, patch):
        size = self.patch_size * self.patch_display_scale
        temp_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        pat_color = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        temp_disp = cv2.resize(temp_color, (size, size), interpolation=cv2.INTER_NEAREST)
        pat_disp = cv2.resize(pat_color, (size, size), interpolation=cv2.INTER_NEAREST)
        t_h, t_w = temp_disp.shape[:2]
        p_h, p_w = pat_disp.shape[:2]
        t_qimg = QImage(temp_disp.data, t_w, t_h, 3*t_w, QImage.Format_BGR888)
        p_qimg = QImage(pat_disp.data, p_w, p_h, 3*p_w, QImage.Format_BGR888)
        self.template_view.setPixmap(QPixmap.fromImage(t_qimg))
        self.match_view.setPixmap(QPixmap.fromImage(p_qimg))

    def draw_epipolar_and_marker(self, a, b, c, best_pt):
        pixmap = self.right_pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        h, w = self.right_gray.shape
        y0 = int(round(-(a*0 + c) / b)) if b != 0 else 0
        y1 = int(round(-(a*w + c) / b)) if b != 0 else h
        painter.drawLine(0, y0, w, y1)
        x2, y2 = best_pt
        painter.drawLine(x2-5, y2, x2+5, y2)
        painter.drawLine(x2, y2-5, x2, y2+5)
        painter.end()
        self.right_label.setPixmap(pixmap)

if __name__ == '__main__':
    left = sys.argv[1] if len(sys.argv) > 1 else None
    right = sys.argv[2] if len(sys.argv) > 2 else None
    app = QApplication(sys.argv)
    window = StereoEpipolarApp(left, right)
    window.show()
    sys.exit(app.exec_())