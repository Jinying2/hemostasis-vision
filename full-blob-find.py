import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
import matplotlib.cm as cm
from utils import *

# ==========================================
# CONFIG
# ==========================================
VIDEO_L = "raw_left_1775254529.mp4"
VIDEO_R = "raw_right_1775254529.mp4"

BLOB_PARAMS = {
    "thresh_block": 21,
    "thresh_C":     10,
    "min_area":     8,
    "max_area":     400,
    "min_circ":     40,
    "nms_radius":   10,
    "use_thresh":   1,
    "use_tophat":   1,
    "use_dog":      1,
    "use_blob_det": 1,
    "tophat_k":     9,
    "dog_sig1":     10,
    "dog_sig2":     25,
}


FB_THRESH     = 2.0
REPROJ_THRESH = 50.0   # px — set wide to diagnose; tighten after seeing values

# ==========================================
# 1. CALIBRATION
# ==========================================
print("📂 Loading Calibration...")
data  = np.load("fisheye_stereo_calib.npz")
K_l, D_l = data['K_l'], data['D_l']
K_r, D_r = data['K_r'], data['D_r']

R_ext = np.array([[ 0.74054941,  0.01090925,  0.67191336],
                  [ 0.01399448,  0.99940102, -0.03165039],
                  [-0.67185618,  0.03284175,  0.73995317]])
T_ext = np.array([[-51.65971666], [-1.49079248], [19.92699316]])
P_l   = np.hstack((np.eye(3), np.zeros((3, 1))))
P_r   = np.hstack((R_ext, T_ext))
rvec_l = np.zeros((3, 1), dtype=np.float64)
tvec_l = np.zeros((3, 1), dtype=np.float64)
rvec_r, _ = cv2.Rodrigues(R_ext)
tvec_r = T_ext.astype(np.float64)

# ==========================================
# 2. READ FRAME 0
# ==========================================
cap_l = cv2.VideoCapture(VIDEO_L)
cap_r = cv2.VideoCapture(VIDEO_R)
ret_l, frame_l = cap_l.read()
ret_r, frame_r = cap_r.read()

if not ret_l or not ret_r:
    raise RuntimeError("Could not read first frame — check video paths")
print(f"✅ Frame 0: {frame_l.shape[1]}x{frame_l.shape[0]}")

# ==========================================
# 3. BLOB DETECTION + GRID MATCHING
# ==========================================
print("🔍 Detecting initial blobs...")
pts_l = find_blobs(frame_l, BLOB_PARAMS)
pts_r = find_blobs(frame_r, BLOB_PARAMS)
print(f"   Left: {len(pts_l)} | Right: {len(pts_r)}")

print("🧅 Extracting grid...")
matched_l, matched_r, layer_labels = extract_full_grid(pts_l, pts_r)
if matched_l is None:
    raise RuntimeError("Grid extraction failed on Frame 0.")
print(f"✅ Matched {len(matched_l)} point pairs")

base_area = cv2.contourArea(get_clockwise_corners(matched_l[:4]))

# ==========================================
# 4. TRIANGULATE FRAME 0
# ==========================================
print("📐 Triangulating frame 0...")
norm_l_m = cv2.fisheye.undistortPoints(matched_l.reshape(-1,1,2), K_l, D_l).reshape(-1,2)
norm_r_m = cv2.fisheye.undistortPoints(matched_r.reshape(-1,1,2), K_r, D_r).reshape(-1,2)
p4d = cv2.triangulatePoints(P_l, P_r, norm_l_m.T, norm_r_m.T)
pts_3d = (p4d[:3] / (p4d[3] + 1e-8)).T
print(f"✅ Reconstructed {len(pts_3d)} 3D points.")

# ==========================================
# 5. SETUP TRACKING
# ==========================================
cap_l.release(); cap_r.release()
cap_l = cv2.VideoCapture(VIDEO_L)
cap_r = cv2.VideoCapture(VIDEO_R)
ret_l, old_frame_l = cap_l.read()
ret_r, old_frame_r = cap_r.read()

old_gray_l = cv2.cvtColor(old_frame_l, cv2.COLOR_BGR2GRAY)
old_gray_r = cv2.cvtColor(old_frame_r, cv2.COLOR_BGR2GRAY)

p0_l = matched_l.reshape(-1, 1, 2).astype(np.float32)
p0_r = matched_r.reshape(-1, 1, 2).astype(np.float32)

lk_params = dict(winSize=(45, 45),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

fig, ax_l, ax_r, ax_3d, ax_pts, ax_err = make_figure()
layer_colors = cm.rainbow(np.linspace(0, 1, 8))
point_colors = [layer_colors[lbl] for lbl in layer_labels]

frame_count = 1
hist_frames, hist_pts, hist_err = [], [], []
mean_err = 0.0

# ==========================================
# 6. MAIN LOOP
# ==========================================
print("🎥 Starting tracking...")
while True:
    ret_l, frame_l = cap_l.read()
    ret_r, frame_r = cap_r.read()
    if not ret_l or not ret_r: break

    frame_gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    # ── Auto-recovery (v1 tol = 15%) ──
    if len(p0_l) < 225:
        new_pts_l = find_blobs(frame_l, BLOB_PARAMS)
        new_pts_r = find_blobs(frame_r, BLOB_PARAMS)
        if len(new_pts_l) >= 225 and len(new_pts_r) >= 225:
            current_corners = get_clockwise_corners(new_pts_l)
            current_area = cv2.contourArea(current_corners)
            if abs(current_area - base_area) / base_area < 0.15:
                m_l, m_r, l_lbls = extract_full_grid(new_pts_l, new_pts_r)
                if m_l is not None and len(m_l) == 225:
                    print(f"🔄 Frame {frame_count}: RESET!")
                    p0_l = m_l.reshape(-1, 1, 2)
                    p0_r = m_r.reshape(-1, 1, 2)
                    point_colors = [layer_colors[lbl] for lbl in l_lbls]
                    old_gray_l = frame_gray_l.copy()
                    old_gray_r = frame_gray_r.copy()
                    frame_count += 1
                    continue

    # ── Forward-Backward LK (with NaN protection) ──
    p1_l, st_l, _ = cv2.calcOpticalFlowPyrLK(old_gray_l, frame_gray_l, p0_l, None, **lk_params)
    p1_r, st_r, _ = cv2.calcOpticalFlowPyrLK(old_gray_r, frame_gray_r, p0_r, None, **lk_params)

    p1_l = np.where(np.isfinite(p1_l), p1_l, p0_l)
    p1_r = np.where(np.isfinite(p1_r), p1_r, p0_r)

    p0_rev_l, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray_l, old_gray_l, p1_l, None, **lk_params)
    p0_rev_r, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray_r, old_gray_r, p1_r, None, **lk_params)

    dist_l = np.linalg.norm((p0_l - p0_rev_l).reshape(-1, 2), axis=1)
    dist_r = np.linalg.norm((p0_r - p0_rev_r).reshape(-1, 2), axis=1)
    good_mask = (st_l.flatten() == 1) & (st_r.flatten() == 1) & (dist_l < FB_THRESH) & (dist_r < FB_THRESH)

    good_new_l = p1_l[good_mask]
    good_new_r = p1_r[good_mask]
    current_colors = np.array(point_colors)[good_mask]

    # ── Triangulate + Reprojection Filter ──
    pts_3d = np.zeros((0, 3))
    mean_err = 0.0
    if len(good_new_l) > 0:
        norm_l = cv2.fisheye.undistortPoints(good_new_l.reshape(-1,1,2), K_l, D_l).reshape(-1,2)
        norm_r = cv2.fisheye.undistortPoints(good_new_r.reshape(-1,1,2), K_r, D_r).reshape(-1,2)
        p4d = cv2.triangulatePoints(P_l, P_r, norm_l.T, norm_r.T)
        pts_3d_all = (p4d[:3] / (p4d[3] + 1e-8)).T

        obj_pts = pts_3d_all.reshape(-1, 1, 3).astype(np.float64)
        rp_l, _ = cv2.fisheye.projectPoints(obj_pts, rvec_l, tvec_l, K_l, D_l)
        rp_r, _ = cv2.fisheye.projectPoints(obj_pts, rvec_r, tvec_r, K_r, D_r)
        err_l = np.linalg.norm(rp_l.reshape(-1,2) - good_new_l.reshape(-1,2), axis=1)
        err_r = np.linalg.norm(rp_r.reshape(-1,2) - good_new_r.reshape(-1,2), axis=1)
        reproj_mask = (err_l < REPROJ_THRESH) & (err_r < REPROJ_THRESH)

        pts_3d         = pts_3d_all[reproj_mask]
        current_colors = current_colors[reproj_mask]
        good_new_l     = good_new_l[reproj_mask]
        good_new_r     = good_new_r[reproj_mask]
        mean_err       = float(np.mean(np.maximum(err_l, err_r)[reproj_mask])) if reproj_mask.any() else 0.0

    hist_frames.append(frame_count)
    hist_pts.append(len(good_new_l))
    hist_err.append(mean_err)

    alive = update_figure(fig, ax_l, ax_r, ax_3d, ax_pts, ax_err,
                          frame_l, frame_r, pts_3d, good_new_l, good_new_r,
                          current_colors, hist_frames, hist_pts, hist_err, frame_count)
    if not alive:
        break

    old_gray_l = frame_gray_l.copy()
    old_gray_r = frame_gray_r.copy()
    p0_l = good_new_l.reshape(-1, 1, 2)
    p0_r = good_new_r.reshape(-1, 1, 2)
    point_colors = list(current_colors)
    frame_count += 1

cap_l.release()
cap_r.release()
plt.ioff()
plt.show()
