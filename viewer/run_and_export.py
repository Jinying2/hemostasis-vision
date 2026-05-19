"""
run_and_export.py
─────────────────
Runs the stereo-blob + LK pipeline and writes viewer/stereo_experiment.json,
which index.html (Three.js) reads for shell rendering.

Usage
-----
    cd hemostasis
    python viewer/run_and_export.py
    python viewer/run_and_export.py left.mp4 right.mp4
"""

import os, sys, json
import cv2
import numpy as np
from scipy.spatial import Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import find_blobs, extract_full_grid, get_clockwise_corners

BLOB_PARAMS = {
    "thresh_block": 21, "thresh_C": 10, "min_area": 8, "max_area": 400,
    "min_circ": 40, "nms_radius": 10, "use_thresh": 1, "use_tophat": 1,
    "use_dog": 1, "use_blob_det": 1, "tophat_k": 9, "dog_sig1": 10, "dog_sig2": 25,
}

# ── config ────────────────────────────────────────────────────────────────────
VIDEO_L = os.path.join(ROOT, "raw_left_1775254529.mp4")
VIDEO_R = os.path.join(ROOT, "raw_right_1775254529.mp4")
CALIB   = os.path.join(ROOT, "fisheye_stereo_calib.npz")
OUT     = os.path.join(ROOT, "viewer", "stereo_experiment.json")

if len(sys.argv) == 3:
    VIDEO_L, VIDEO_R = sys.argv[1], sys.argv[2]

# ── calibration ───────────────────────────────────────────────────────────────
print("📂  Loading calibration...")
cal   = np.load(CALIB)
K_l, D_l = cal["K_l"], cal["D_l"]
K_r, D_r = cal["K_r"], cal["D_r"]
R_ext = np.array([[ 0.74054941,  0.01090925,  0.67191336],
                  [ 0.01399448,  0.99940102, -0.03165039],
                  [-0.67185618,  0.03284175,  0.73995317]])
T_ext = np.array([[-51.65971666], [-1.49079248], [19.92699316]])
P_l   = np.hstack((np.eye(3), np.zeros((3, 1))))
P_r   = np.hstack((R_ext, T_ext))


def triangulate(pts_l, pts_r):
    nl = cv2.fisheye.undistortPoints(pts_l.reshape(-1,1,2), K_l, D_l).reshape(-1,2)
    nr = cv2.fisheye.undistortPoints(pts_r.reshape(-1,1,2), K_r, D_r).reshape(-1,2)
    p4d = cv2.triangulatePoints(P_l, P_r, nl.T, nr.T)
    return (p4d[:3] / (p4d[3] + 1e-8)).T


def build_mesh(pts2d):
    """Delaunay on 2D image coords — same concept as VisuoShell build_mesh."""
    s    = Delaunay(pts2d).simplices
    def me(t):
        a,b,c = pts2d[t[0]],pts2d[t[1]],pts2d[t[2]]
        return max(np.linalg.norm(a-b), np.linalg.norm(b-c), np.linalg.norm(c-a))
    lens = np.array([me(t) for t in s])
    return s[lens < 3.0 * np.median(lens)].tolist()


# ── frame 0 ───────────────────────────────────────────────────────────────────
cap_l = cv2.VideoCapture(VIDEO_L)
cap_r = cv2.VideoCapture(VIDEO_R)
fps   = int(cap_l.get(cv2.CAP_PROP_FPS)) or 30
_, f0_l = cap_l.read()
_, f0_r = cap_r.read()
print(f"✅  Frame 0: {f0_l.shape[1]}×{f0_l.shape[0]}  FPS={fps}")

print("🔍  Blob detection + grid matching...")
matched_l, matched_r, _ = extract_full_grid(
    find_blobs(f0_l, BLOB_PARAMS), find_blobs(f0_r, BLOB_PARAMS))
if matched_l is None:
    sys.exit("Grid matching failed")
print(f"    {len(matched_l)} point pairs")

pts3d_0 = triangulate(matched_l, matched_r)
faces   = build_mesh(matched_l)
print(f"    {len(faces)} triangles")

# ── LK tracking loop ──────────────────────────────────────────────────────────
cap_l.release(); cap_r.release()
cap_l = cv2.VideoCapture(VIDEO_L); cap_r = cv2.VideoCapture(VIDEO_R)
_, ol = cap_l.read(); _, or_ = cap_r.read()
old_gl = cv2.cvtColor(ol,  cv2.COLOR_BGR2GRAY)
old_gr = cv2.cvtColor(or_, cv2.COLOR_BGR2GRAY)

lk   = dict(winSize=(45,45), maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p0_l = matched_l.reshape(-1,1,2).astype(np.float32)
p0_r = matched_r.reshape(-1,1,2).astype(np.float32)
pids = np.arange(len(matched_l), dtype=int)
cur3d = pts3d_0.copy()
base_area  = cv2.contourArea(get_clockwise_corners(matched_l[:4]))
all_frames = [pts3d_0.flatten().round(3).tolist()]
timestamps = [0.0]   # seconds; frame 0 is at t=0

print("🎥  Tracking (Ctrl-C to stop early)...")
fidx = 1
try:
    while True:
        ret_l, fr_l = cap_l.read(); ret_r, fr_r = cap_r.read()
        if not ret_l or not ret_r: break
        gl = cv2.cvtColor(fr_l, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fr_r, cv2.COLOR_BGR2GRAY)

        if len(p0_l) < 225:
            nl = find_blobs(fr_l, BLOB_PARAMS); nr = find_blobs(fr_r, BLOB_PARAMS)
            if len(nl) >= 225 and len(nr) >= 225:
                c = get_clockwise_corners(nl)
                if abs(cv2.contourArea(c) - base_area) / base_area < 0.05:
                    ml, mr, _ = extract_full_grid(nl, nr)
                    if ml is not None and len(ml) == 225:
                        p0_l=ml.reshape(-1,1,2); p0_r=mr.reshape(-1,1,2)
                        pids=np.arange(225,dtype=int)
                        old_gl=gl.copy(); old_gr=gr.copy(); fidx+=1; continue

        p1_l,sl,_ = cv2.calcOpticalFlowPyrLK(old_gl,gl,p0_l,None,**lk)
        p1_r,sr,_ = cv2.calcOpticalFlowPyrLK(old_gr,gr,p0_r,None,**lk)
        pb_l,_,_  = cv2.calcOpticalFlowPyrLK(gl,old_gl,p1_l,None,**lk)
        pb_r,_,_  = cv2.calcOpticalFlowPyrLK(gr,old_gr,p1_r,None,**lk)
        dl   = np.linalg.norm((p0_l-pb_l).reshape(-1,2),axis=1)
        dr   = np.linalg.norm((p0_r-pb_r).reshape(-1,2),axis=1)
        good = (sl.flatten()==1)&(sr.flatten()==1)&(dl<1.0)&(dr<1.0)

        if good.sum() > 0:
            cur3d[pids[good]] = triangulate(p1_l[good].reshape(-1,2),
                                             p1_r[good].reshape(-1,2))
        all_frames.append(cur3d.flatten().round(3).tolist())
        timestamps.append(round(cap_l.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 6))
        if fidx % 100 == 0:
            print(f"    frame {fidx}: {good.sum()}/{len(pids)} tracked")

        old_gl=gl.copy(); old_gr=gr.copy()
        p0_l=p1_l[good]; p0_r=p1_r[good]; pids=pids[good]; fidx+=1

except KeyboardInterrupt:
    print(f"\n⚠️  Stopped at frame {fidx}")

cap_l.release(); cap_r.release()
print(f"✅  {len(all_frames)} frames collected")

payload = {"source":"experiment","fps":fps,"n_points":int(len(matched_l)),
           "faces":faces,"frames":all_frames,"timestamps":timestamps,
           "calib":{"R_ext":R_ext.tolist(),"T_ext":T_ext.flatten().tolist()}}
with open(OUT,"w") as f: json.dump(payload,f)
print(f"💾  Saved → {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
print("👉  Serve:  python -m http.server -d . 8000  → open localhost:8000/viewer/")
