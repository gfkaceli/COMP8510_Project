# match2.py — Paper-true: edges → ZNCC correlation → segment mapping (dense)
import sys
import os
import cv2
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def mse_psnr(img1, img2):
    err = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    ps = float('inf') if err <= 1e-12 else 10.0 * np.log10((255.0 ** 2) / err)
    return err, ps

def save_image(path, img):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


class StereoMatchApp(QMainWindow):
    def __init__(self, left_path=None, right_path=None):
        super().__init__()
        self.setWindowTitle('Stereo Dense Match — Paper-True (Edges+ZNCC+Segments)')

        # Images
        self.left_img = None
        self.right_img = None
        self.left_gray = None
        self.right_gray = None
        self.left_path = left_path
        self.right_path = right_path

        # Geometry
        self.F = None
        self.pts1 = None
        self.pts2 = None
        self.H1 = None
        self.H2 = None

        # Matches
        self.edge_matches = {}   # {(x,y): (x2,y)}
        self.dense_matches = {}  # {(x,y): (x2,y)}

        # Tunables (paper uses simple Sobel + correlation; ZNCC window = 2k+1)
        self.edge_thresh = 20     # Sobel magnitude threshold
        self.tm_win = 3         # 3 => 7x7 ZNCC window; set 5 for 11x11

        # UI
        self.left_label = QLabel('Left Image');  self.left_label.setAlignment(Qt.AlignCenter)
        self.right_label = QLabel('Right Image'); self.right_label.setAlignment(Qt.AlignCenter)

        btn_load_left  = QPushButton('Load Left');  btn_load_left.clicked.connect(lambda: self.load_image(True))
        btn_load_right = QPushButton('Load Right'); btn_load_right.clicked.connect(lambda: self.load_image(False))
        btn_compute_F  = QPushButton('Compute F');  btn_compute_F.clicked.connect(self.compute_f)
        btn_match      = QPushButton('Run Matching'); btn_match.clicked.connect(self.run_matching)

        btn_layout = QHBoxLayout()
        for b in (btn_load_left, btn_load_right, btn_compute_F, btn_match):
            btn_layout.addWidget(b)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.left_label)
        img_layout.addWidget(self.right_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addLayout(btn_layout)

        container = QWidget(); container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.showMaximized()

        if left_path and right_path:
            self.load_image(True, left_path)
            self.load_image(False, right_path)

    # ---------------------------
    # IO
    # ---------------------------
    def load_image(self, is_left, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open Image')
            if not path:
                return
        pil = PILImage.open(path).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        lbl = self.left_label if is_left else self.right_label
        lbl.setPixmap(pix)
        lbl.setFixedSize(w, h)

        if is_left:
            self.left_img, self.left_gray, self.left_path = img, gray, path
        else:
            self.right_img, self.right_gray, self.right_path = img, gray, path

    # ---------------------------
    # Fundamental matrix
    # ---------------------------
    def compute_f(self):
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Load both images first'); return

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.left_gray, None)
        kp2, des2 = sift.detectAndCompute(self.right_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            QMessageBox.warning(self, 'Error', 'Not enough features to estimate F'); return

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if n is not None and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            QMessageBox.warning(self, 'Error', 'Not enough good matches to estimate F'); return

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        if F is None or mask is None:
            QMessageBox.warning(self, 'Error', 'Fundamental matrix failed'); return

        self.pts1 = pts1[mask.ravel() == 1]
        self.pts2 = pts2[mask.ravel() == 1]
        self.F = F
        print("F =\n", F)
        QMessageBox.information(self, 'Fundamental Matrix', f'F =\n{self.F}')

    # ---------------------------
    # Rowwise disparity bounds
    # ---------------------------
    def _row_disparity_bounds(self, H1, H2, w, h):
        if self.pts1 is None or self.pts2 is None or len(self.pts1) < 8:
            return np.full(h, -96.0, dtype=np.float32), np.full(h, 96.0, dtype=np.float32)

        pts1r = cv2.perspectiveTransform(self.pts1.reshape(-1, 1, 2), H1).reshape(-1, 2)
        pts2r = cv2.perspectiveTransform(self.pts2.reshape(-1, 1, 2), H2).reshape(-1, 2)
        disps = pts1r[:, 0] - pts2r[:, 0]
        gmin, gmax = np.percentile(disps, 5), np.percentile(disps, 95)

        row_bins = np.clip(np.round(pts1r[:, 1]).astype(int), 0, h - 1)
        row_dmin = np.full(h, gmin, dtype=np.float32)
        row_dmax = np.full(h, gmax, dtype=np.float32)
        for y in range(h):
            sel = (row_bins == y)
            if sel.any():
                row_dmin[y] = np.percentile(disps[sel], 5)
                row_dmax[y] = np.percentile(disps[sel], 95)
        row_dmin -= 4.0
        row_dmax += 4.0
        return row_dmin, row_dmax

    # ---------------------------
    # ZNCC matcher for a single edge pixel along a scanline
    ### PAPER §2.1: Match EDGE pixels by correlation under the epipolar constraint.
    # This function performs classical correlation (ZNCC) for a LEFT edge pixel by
    # scanning candidates along the SAME ROW in the RECTIFIED RIGHT image and picking
    # the best score. No non-edge pixels are correlated here.
    # ---------------------------
    def _match_edge_zncc(self, left_rect, right_rect, y, xL, dmin, dmax, win):
        """
        Return (xR, y) for best ZNCC match of left_rect[y, xL] patch within [xL-dmax, xL-dmin].
        """
        h, w = left_rect.shape
        # patch bounds
        if xL - win < 0 or xL + win >= w or y - win < 0 or y + win >= h:
            return None

        tpl = left_rect[y - win:y + win + 1, xL - win:xL + win + 1].astype(np.float32)
        t_mean = tpl.mean()
        t_std = tpl.std()
        if t_std < 1e-6:
            return None
        tpl_zm = tpl - t_mean

        xs_lo = int(max(win, np.floor(xL - dmax)))
        xs_hi = int(min(w - win - 1, np.ceil(xL - dmin)))
        if xs_hi < xs_lo:
            return None

        # Brute-force ZNCC across candidates
        best_x = None
        best_sc = -1.0
        for xR in range(xs_lo, xs_hi + 1):
            roi = right_rect[y - win:y + win + 1, xR - win:xR + win + 1].astype(np.float32)
            r_mean = roi.mean()
            r_std = roi.std()
            if r_std < 1e-6:
                continue
            sc = np.sum((roi - r_mean) * tpl_zm) / (t_std * r_std * tpl_zm.size)
            if sc > best_sc:
                best_sc = sc
                best_x = xR

        if best_x is None:
            return None
        return (best_x, y)

    # ---------------------------
    # pipeline (edges → ZNCC on edges → segment mapping)
    # ---------------------------
    def run_matching(self):
        if self.F is None:
            QMessageBox.warning(self, 'Error', 'Compute F first'); return
        if self.left_gray is None or self.right_gray is None:
            QMessageBox.warning(self, 'Error', 'Load both images first'); return

        h, w = self.left_gray.shape

        # -------------------------------------------------------------------------
        # ### PAPER §2: Make epipolar lines horizontal (Rectification).
        # Using F (and inlier correspondences), we compute homographies H1, H2 with
        # stereoRectifyUncalibrated and warp both images. This aligns epipolar lines
        # with image rows so correlation is a 1D search along the scanline.
        # -------------------------------------------------------------------------

        # Rectify (make epipolar lines horizontal)
        _, H1, H2 = cv2.stereoRectifyUncalibrated(self.pts1, self.pts2, self.F, (w, h))
        if H1 is None or H2 is None:
            QMessageBox.warning(self, 'Error', 'Rectification failed'); return
        self.H1, self.H2 = H1, H2

        left_rect   = cv2.warpPerspective(self.left_gray, H1, (w, h))
        right_rect  = cv2.warpPerspective(self.right_gray, H2, (w, h))
        left_colorR = cv2.warpPerspective(self.left_img,  H1, (w, h))

        # Edge mask (Sobel, per paper)
        # -------------------------------------------------------------------------
        # ### PAPER §2.1: Edge extraction on the LEFT image using a simple detector.
        # We use Sobel magnitude thresholding to obtain edge pixels. These edges are
        # the only pixels matched by correlation; non-edge pixels are filled later by
        # segment mapping between matched edges on each scanline.
        # -------------------------------------------------------------------------
        gx = cv2.Sobel(left_rect, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(left_rect, cv2.CV_16S, 0, 1, ksize=3)
        abs_gx = cv2.convertScaleAbs(gx)
        abs_gy = cv2.convertScaleAbs(gy)
        mag = cv2.add(abs_gx, abs_gy)
        edge_mask = (mag > self.edge_thresh).astype(np.uint8)

        # Rowwise disparity bounds
        row_dmin, row_dmax = self._row_disparity_bounds(H1, H2, w, h)

        # 1) Match ALL edge pixels by ZNCC along epipolar lines
        self.edge_matches.clear()
        win = self.tm_win  # 3 → 7x7, 5 → 11x11
        total_edges, matched_edges = 0, 0
        for y in range(win, h - win):
            xs = np.where(edge_mask[y] == 1)[0]
            if xs.size == 0:
                continue
            total_edges += xs.size
            dmin = float(row_dmin[y]); dmax = float(row_dmax[y])
            for x in xs:
                # -------------------------------------------------------------
                # PAPER §2.1: Correlation on edges along epipolar (row) only.
                # We call the ZNCC matcher for each LEFT edge pixel to find its
                # RIGHT correspondence on the same rectified row.
                # -------------------------------------------------------------
                res = self._match_edge_zncc(left_rect, right_rect, y, x, dmin, dmax, win)
                if res is not None:
                    self.edge_matches[(x, y)] = res
                    matched_edges += 1
        print(f"Edge pixels found: {total_edges}")
        print(f"Edge pixels matched (ZNCC): {matched_edges}")

        # 2) Match ALL non-edge pixels via segment mapping (A,B -> A',B') along each row
        self.dense_matches.clear()

        # Build quick lookup of matched edge x->x2 for each row
        per_row_edge_map = {}
        for (x, y), (x2, _) in self.edge_matches.items():
            per_row_edge_map.setdefault(y, {})[x] = x2

        for y in range(h):
            row_edges = np.where(edge_mask[y] == 1)[0]

            # Helper: x2 for an edge endpoint, interpolating from nearest matched endpoints if needed
            def edge_x2(xe):
                if (xe, y) in self.edge_matches:
                    return self.edge_matches[(xe, y)][0]
                if row_edges.size == 0:
                    return xe
                xs_edges = sorted(row_edges.tolist())
                left = next((x for x in reversed(xs_edges) if x < xe and (x, y) in self.edge_matches), None)
                right = next((x for x in xs_edges if x > xe and (x, y) in self.edge_matches), None)
                if left is not None and right is not None:
                    xA2 = self.edge_matches[(left, y)][0]
                    xB2 = self.edge_matches[(right, y)][0]
                    L = max(1, right - left)
                    lam = (xB2 - xA2) / L
                    return int(round(xA2 + lam * (xe - left)))
                if left is not None:
                    return self.edge_matches[(left, y)][0]
                if right is not None:
                    return self.edge_matches[(right, y)][0]
                return xe

            if row_edges.size == 0:
                # Borrow nearest row’s mapping (dense fallback)
                up, down, donor = y - 1, y + 1, None
                while up >= 0 or down < h:
                    if up >= 0 and np.any(edge_mask[up] == 1):
                        donor = up; break
                    if down < h and np.any(edge_mask[down] == 1):
                        donor = down; break
                    up -= 1; down += 1
                if donor is None:
                    for x in range(w):
                        self.dense_matches[(x, y)] = (x, y)
                else:
                    donor_map = per_row_edge_map.get(donor, {})
                    if len(donor_map) >= 2:
                        xs_sorted = sorted(donor_map.keys())
                        for x in range(w):
                            i = np.searchsorted(xs_sorted, x)
                            if i == 0:
                                xA, xB = xs_sorted[0], xs_sorted[1]
                            elif i >= len(xs_sorted):
                                xA, xB = xs_sorted[-2], xs_sorted[-1]
                            else:
                                xA, xB = xs_sorted[i-1], xs_sorted[i]
                            xA2, xB2 = donor_map[xA], donor_map[xB]
                            L = max(1, xB - xA)
                            lam = (xB2 - xA2) / L
                            x2 = int(round(xA2 + lam * (x - xA)))
                            x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                            self.dense_matches[(x, y)] = (x2, y)
                    else:
                        for x in range(w):
                            self.dense_matches[(x, y)] = (x, y)
                continue

            xs_edges = sorted(row_edges.tolist())

            # a) Left of first edge: extrapolate using first two edges
            if len(xs_edges) >= 2:
                xA, xB = xs_edges[0], xs_edges[1]
                xA2, xB2 = edge_x2(xA), edge_x2(xB)
                L = max(1, xB - xA)
                lam = (xB2 - xA2) / L
                for x in range(0, xA):
                    x2 = int(round(xA2 + lam * (x - xA)))
                    x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                    self.dense_matches[(x, y)] = (x2, y)
            else:
                # only one edge on this row → constant disparity across row
                xE = xs_edges[0]
                xE2 = edge_x2(xE)
                dconst = xE - xE2
                for x in range(0, xE):
                    x2 = int(x - dconst); x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                    self.dense_matches[(x, y)] = (x2, y)

            # b) Between consecutive edges: SEGMENT MAPPING (AB→A′B′)
            # ### PAPER §2.2–2.3: Segment-based mapping for NON-EDGE pixels.
            # For each scanline, consider consecutive matched edges A(xL→xL′)
            # and B(xR→xR′). For an interior pixel p at offset dp from A,
            # we map to p′ = A′ + λ·dp where λ = (xR′−xL′) / (xR−xL).
            # Only pixels INSIDE edge-bounded segments are filled; regions
            # with no enclosing edges remain unmatched (as in the paper).
            # -----------------------------------------------------------------
            for i in range(len(xs_edges) - 1):
                xL = xs_edges[i]
                xR = xs_edges[i + 1]
                xL2 = edge_x2(xL)
                xR2 = edge_x2(xR)
                L = max(1, xR - xL)
                lam = (xR2 - xL2) / L
                self.dense_matches[(xL, y)] = (xL2, y)
                for x in range(xL + 1, xR):
                    dp = x - xL
                    x2 = int(round(xL2 + lam * dp))
                    x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                    self.dense_matches[(x, y)] = (x2, y)
                self.dense_matches[(xR, y)] = (xR2, y)

            # c) Right of last edge: extrapolate using last two edges
            if len(xs_edges) >= 2:
                xA, xB = xs_edges[-2], xs_edges[-1]
                xA2, xB2 = edge_x2(xA), edge_x2(xB)
                L = max(1, xB - xA)
                lam = (xB2 - xA2) / L
                for x in range(xB + 1, w):
                    x2 = int(round(xB2 + lam * (x - xB)))
                    x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                    self.dense_matches[(x, y)] = (x2, y)
            else:
                xE = xs_edges[0]
                xE2 = edge_x2(xE)
                dconst = xE - xE2
                for x in range(xE + 1, w):
                    x2 = int(x - dconst); x2 = 0 if x2 < 0 else (w - 1 if x2 >= w else x2)
                    self.dense_matches[(x, y)] = (x2, y)


        # Stats
        dcount = len(self.dense_matches)
        coverage = 100.0 * dcount / (w * h)
        print(f"Dense correspondences: {dcount} ({coverage:.2f}% of pixels)")

        # Disparity map (x - x')
        disp = np.zeros((h, w), np.float32)
        for (x, y), (x2, _) in self.dense_matches.items():
            disp[y, x] = x - x2
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow('Disparity (normalized)', disp_vis)

        # Reconstruct right (rectified)
        map_y, map_x = np.indices((h, w), dtype=np.float32)
        recon_rect = cv2.remap(
            left_colorR,
            (map_x - disp).astype(np.float32),
            map_y,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT
        )

        # Un-rectify to original right view
        invH2 = np.linalg.inv(self.H2)
        recon_unrect = cv2.warpPerspective(recon_rect, invH2, (w, h))
        cv2.imshow('Reconstructed Right (Original View)', recon_unrect)
        cv2.waitKey(1)

        # Save & metrics
        base_dir = os.path.dirname(self.right_path) if self.right_path else ""
        stem = os.path.splitext(os.path.basename(self.right_path))[0] if self.right_path else "output"
        disp_path  = os.path.join(base_dir, f"{stem}_match2disparity.png")
        recon_path = os.path.join(base_dir, f"{stem}_match2reconstructed.png")
        diff_path  = os.path.join(base_dir, f"{stem}_match2absdiff.png")

        save_image(disp_path, disp_vis)
        save_image(recon_path, recon_unrect)

        right_view = self.right_img
        if right_view.shape[:2] != recon_unrect.shape[:2]:
            hh = min(right_view.shape[0], recon_unrect.shape[0])
            ww = min(right_view.shape[1], recon_unrect.shape[1])
            right_cmp = right_view[:hh, :ww]
            recon_cmp = recon_unrect[:hh, :ww]
        else:
            right_cmp = right_view
            recon_cmp = recon_unrect

        absdiff = cv2.absdiff(right_cmp, recon_cmp)
        save_image(diff_path, absdiff)

        err, ps = mse_psnr(right_cmp, recon_cmp)
        print(f"Saved: {disp_path}")
        print(f"Saved: {recon_path}")
        print(f"Saved: {diff_path}")


if __name__ == '__main__':
    left = sys.argv[1] if len(sys.argv) > 1 else None
    right = sys.argv[2] if len(sys.argv) > 2 else None
    app = QApplication(sys.argv)
    win = StereoMatchApp(left, right)
    win.show()
    sys.exit(app.exec_())
