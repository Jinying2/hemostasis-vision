"""
utils.py
────────
Shared functions for the stereo-blob tracking pipeline.
Import with: from utils import *
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Blob Detection ────────────────────────────────────────────────────────

def extract_centroids(binary, min_a, max_a, min_circ_pct):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_a < area < max_a): continue
        if min_circ_pct > 0:
            perim = cv2.arcLength(c, True)
            if perim == 0: continue
            if (4 * np.pi * area / perim**2) < min_circ_pct / 100.0: continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append([M["m10"]/M["m00"], M["m01"]/M["m00"]])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0,2), dtype=np.float32)


def nms_merge(all_pts, radius):
    if len(all_pts) == 0: return np.zeros((0,2), dtype=np.float32)
    used = np.zeros(len(all_pts), dtype=bool)
    merged = []
    for i in range(len(all_pts)):
        if used[i]: continue
        dists = np.linalg.norm(all_pts - all_pts[i], axis=1)
        cluster = np.where(dists < radius)[0]
        used[cluster] = True
        merged.append(np.mean(all_pts[cluster], axis=0))
    return np.array(merged, dtype=np.float32)


def find_blobs(frame, p):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_a = max(p["min_area"], 1)
    max_a = max(p["max_area"], min_a + 1)
    min_c = p["min_circ"]
    all_pts = []

    if p["use_thresh"]:
        block = max(p["thresh_block"] if p["thresh_block"] % 2 == 1 else p["thresh_block"]+1, 3)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, p["thresh_C"])
        r = extract_centroids(thresh, min_a, max_a, min_c)
        if len(r): all_pts.append(r)

    if p["use_tophat"]:
        k = max(p["tophat_k"] if p["tophat_k"] % 2 == 1 else p["tophat_k"]+1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        r = extract_centroids(thresh, min_a, max_a, min_c)
        if len(r): all_pts.append(r)

    if p["use_dog"]:
        g1 = cv2.GaussianBlur(gray.astype(np.float32), (0,0), p["dog_sig1"]/10.0)
        g2 = cv2.GaussianBlur(gray.astype(np.float32), (0,0), p["dog_sig2"]/10.0)
        dog_norm = cv2.normalize(g2 - g1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        r = extract_centroids(thresh, min_a, max_a, min_c)
        if len(r): all_pts.append(r)

    if p["use_blob_det"]:
        bp = cv2.SimpleBlobDetector_Params()
        bp.filterByArea = True;        bp.minArea = float(min_a); bp.maxArea = float(max_a)
        bp.filterByColor = True;       bp.blobColor = 0
        bp.filterByCircularity = True; bp.minCircularity = 0.3
        bp.filterByConvexity = True;   bp.minConvexity = 0.7
        bp.filterByInertia = True;     bp.minInertiaRatio = 0.3
        kps = cv2.SimpleBlobDetector_create(bp).detect(gray)
        if kps:
            r = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
            all_pts.append(r)

    if not all_pts: return np.zeros((0,2), dtype=np.float32)
    return nms_merge(np.vstack(all_pts), p["nms_radius"])


# ── Grid Matching ─────────────────────────────────────────────────────────

def get_clockwise_corners(pts):
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmax(diff)],
        pts[np.argmax(s)],
        pts[np.argmin(diff)]
    ], dtype=np.float32)


def remove_point(pool, target_pt):
    idx = np.argmin(np.linalg.norm(pool - target_pt, axis=1))
    return np.delete(pool, idx, axis=0)


def extract_edge_walk(pool, pt_start, pt_end, num_internal_pts):
    edge_pts = []
    curr = pt_start
    step_vec = (pt_end - pt_start) / (num_internal_pts + 1)
    for _ in range(num_internal_pts):
        target = curr + step_vec
        idx = np.argmin(np.linalg.norm(pool - target, axis=1))
        best_pt = pool[idx]
        edge_pts.append(best_pt)
        pool = np.delete(pool, idx, axis=0)
        curr = best_pt
        remaining_steps = num_internal_pts - len(edge_pts) + 1
        step_vec = (pt_end - curr) / remaining_steps
    return np.array(edge_pts), pool


def extract_full_grid(p_l, p_r):
    grid_sizes = [15, 13, 11, 9, 7, 5, 3, 1]
    m_l, m_r, l_labels = [], [], []
    pool_l, pool_r = np.copy(p_l), np.copy(p_r)
    try:
        for layer_idx, g_size in enumerate(grid_sizes):
            if g_size == 1:
                if len(pool_l) > 0 and len(pool_r) > 0:
                    c_l = np.mean(pool_l, axis=0)
                    c_r = np.mean(pool_r, axis=0)
                    m_l.append(pool_l[np.argmin(np.linalg.norm(pool_l - c_l, axis=1))])
                    m_r.append(pool_r[np.argmin(np.linalg.norm(pool_r - c_r, axis=1))])
                    l_labels.append(layer_idx)
                break
            corners_l = get_clockwise_corners(pool_l)
            corners_r = get_clockwise_corners(pool_r)
            for cl, cr in zip(corners_l, corners_r):
                m_l.append(cl); m_r.append(cr); l_labels.append(layer_idx)
                pool_l = remove_point(pool_l, cl); pool_r = remove_point(pool_r, cr)
            num_internal = g_size - 2
            for start_idx, end_idx in [(0,1),(1,2),(2,3),(3,0)]:
                if num_internal > 0:
                    edges_l, pool_l = extract_edge_walk(pool_l, corners_l[start_idx], corners_l[end_idx], num_internal)
                    edges_r, pool_r = extract_edge_walk(pool_r, corners_r[start_idx], corners_r[end_idx], num_internal)
                    for el, er in zip(edges_l, edges_r):
                        m_l.append(el); m_r.append(er); l_labels.append(layer_idx)
        return np.array(m_l, dtype=np.float32), np.array(m_r, dtype=np.float32), np.array(l_labels)
    except Exception:
        return None, None, None


# ── Visualization ─────────────────────────────────────────────────────────

def make_figure():
    """Create the shared 5-panel dark figure.
    Returns (fig, ax_l, ax_r, ax_3d, ax_pts, ax_err)."""
    plt.ion()
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#0d0d0d')
    gs = GridSpec(2, 6, figure=fig, height_ratios=[3, 1.5], hspace=0.45, wspace=0.5)
    ax_l   = fig.add_subplot(gs[0, 0:2])
    ax_r   = fig.add_subplot(gs[0, 2:4])
    ax_3d  = fig.add_subplot(gs[0, 4:6], projection='3d')
    ax_pts = fig.add_subplot(gs[1, 0:3])
    ax_err = fig.add_subplot(gs[1, 3:6])
    for ax in [ax_l, ax_r, ax_3d, ax_pts, ax_err]:
        ax.set_facecolor('#0d0d0d')
    return fig, ax_l, ax_r, ax_3d, ax_pts, ax_err


def update_figure(fig, ax_l, ax_r, ax_3d, ax_pts, ax_err,
                  frame_l, frame_r, pts_3d, good_new_l, good_new_r,
                  colors, hist_frames, hist_pts, hist_err, frame_count):
    """Redraw all 5 panels for the current frame.
    Returns False if the window was closed (use as exit signal)."""

    # 3D scatter
    ax_3d.clear(); ax_3d.set_facecolor('#0d0d0d')
    if len(pts_3d) > 0:
        ax_3d.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2],
                      s=25, c=colors, alpha=0.9, edgecolors='none')
        cx, cy, cz = np.mean(pts_3d, axis=0)
        r = 25
        ax_3d.set_xlim(cx-r, cx+r); ax_3d.set_ylim(cy-r, cy+r); ax_3d.set_zlim(cz-r, cz+r)
    ax_3d.set_xlabel('X (mm)', color='white', fontsize=8)
    ax_3d.set_ylabel('Y (mm)', color='white', fontsize=8)
    ax_3d.set_zlabel('Z (mm)', color='white', fontsize=8)
    ax_3d.set_title('3D — Live Reconstruction', color='white', fontsize=9)
    ax_3d.tick_params(colors='white', labelsize=6)
    ax_3d.invert_yaxis()

    # 2D left
    ax_l.clear(); ax_l.set_facecolor('#0d0d0d')
    ax_l.imshow(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
    if len(good_new_l) > 0:
        pts2d = good_new_l.reshape(-1, 2)
        ax_l.scatter(pts2d[:, 0], pts2d[:, 1], s=8, c=colors, linewidths=0)
    ax_l.set_title("Left — LK Tracking", color='white', fontsize=9)
    ax_l.axis('off')

    # 2D right
    ax_r.clear(); ax_r.set_facecolor('#0d0d0d')
    ax_r.imshow(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
    if len(good_new_r) > 0:
        pts2d = good_new_r.reshape(-1, 2)
        ax_r.scatter(pts2d[:, 0], pts2d[:, 1], s=8, c=colors, linewidths=0)
    ax_r.set_title("Right — LK Tracking", color='white', fontsize=9)
    ax_r.axis('off')

    fig.suptitle(f"Frame {frame_count}  |  tracked: {len(good_new_l)}", color='white', fontsize=11)

    # tracked pts time series
    ax_pts.clear(); ax_pts.set_facecolor('#0d0d0d')
    ax_pts.plot(hist_frames, hist_pts, color='#00e676', linewidth=1)
    ax_pts.set_ylim(0, 230)
    ax_pts.set_ylabel('tracked pts', color='white', fontsize=8)
    ax_pts.tick_params(colors='white', labelsize=7)
    for sp in ax_pts.spines.values(): sp.set_color('#444')

    # reproj err time series
    ax_err.clear(); ax_err.set_facecolor('#0d0d0d')
    ax_err.plot(hist_frames, hist_err, color='#ff9100', linewidth=1)
    ax_err.set_ylabel('reproj err (px)', color='white', fontsize=8)
    ax_err.tick_params(colors='white', labelsize=7)
    for sp in ax_err.spines.values(): sp.set_color('#444')

    plt.draw()
    plt.pause(0.001)
    return plt.fignum_exists(fig.number)

