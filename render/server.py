"""
render/server.py
────────────────
Local Flask server for the Hemostasis renderer.

- Runs the stereo reconstruction pipeline in a background thread
- Streams progress via SSE
- Saves result to viewer/stereo_experiment.json (same as run_and_export.py)
- Serves render/index.html

Usage:
    cd hemostasis
    python render/server.py
    # Open http://localhost:8000/render/
"""

import ast
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback

from flask import Flask, Response, jsonify, redirect, request, send_file, send_from_directory

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(ROOT, "fisheye_stereo_calib.npz")
OUT = os.path.join(ROOT, "viewer", "stereo_experiment.json")
DEFAULT_L = os.path.join(ROOT, "raw_left_1775254529.mp4")
DEFAULT_R = os.path.join(ROOT, "raw_right_1775254529.mp4")

# ── Load blob-pipeline functions from full-blob-find.py (no side-effects) ────
def _load_pipeline():
    with open(os.path.join(ROOT, "full-blob-find.py"), encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "BLOB_PARAMS"
                        for t in n.targets))]
    mod = ast.Module(body=keep, type_ignores=[])
    ns  = {}
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    exec(compile(mod, "full-blob-find.py", "exec"), ns)
    return ns

PIPELINE_DEPS_OK = all(dep is not None for dep in (cv2, np, Delaunay))
PIPELINE_IMPORT_ERROR = None if PIPELINE_DEPS_OK else "Missing pipeline dependencies: install numpy, scipy, and opencv-python."

if PIPELINE_DEPS_OK:
    _p = _load_pipeline()
    find_blobs = _p["find_blobs"]
    extract_full_grid = _p["extract_full_grid"]
    get_clockwise_corners = _p["get_clockwise_corners"]
    BLOB_PARAMS = _p["BLOB_PARAMS"]
else:
    find_blobs = None
    extract_full_grid = None
    get_clockwise_corners = None
    BLOB_PARAMS = None

# ── Calibration ───────────────────────────────────────────────────────────────
if PIPELINE_DEPS_OK:
    cal = np.load(CALIB)
    K_l, D_l = cal["K_l"], cal["D_l"]
    K_r, D_r = cal["K_r"], cal["D_r"]
    R_ext = np.array([[ 0.74054941,  0.01090925,  0.67191336],
                      [ 0.01399448,  0.99940102, -0.03165039],
                      [-0.67185618,  0.03284175,  0.73995317]])
    T_ext = np.array([[-51.65971666], [-1.49079248], [19.92699316]])
    P_l   = np.hstack((np.eye(3), np.zeros((3, 1))))
    P_r   = np.hstack((R_ext, T_ext))
else:
    K_l = D_l = K_r = D_r = R_ext = T_ext = P_l = P_r = None

def triangulate(pts_l, pts_r):
    nl = cv2.fisheye.undistortPoints(pts_l.reshape(-1,1,2), K_l, D_l).reshape(-1,2)
    nr = cv2.fisheye.undistortPoints(pts_r.reshape(-1,1,2), K_r, D_r).reshape(-1,2)
    p4d = cv2.triangulatePoints(P_l, P_r, nl.T, nr.T)
    return (p4d[:3] / (p4d[3] + 1e-8)).T

def build_mesh(pts2d):
    s    = Delaunay(pts2d).simplices
    def me(t):
        a,b,c = pts2d[t[0]], pts2d[t[1]], pts2d[t[2]]
        return max(np.linalg.norm(a-b), np.linalg.norm(b-c), np.linalg.norm(c-a))
    lens = np.array([me(t) for t in s])
    return s[lens < 3.0 * np.median(lens)].tolist()

# ── Pipeline (runs in background thread, posts events to a queue) ─────────────
def run_pipeline(path_l, path_r, q):
    def emit(step, pct):
        q.put({"type": "progress", "step": step, "pct": pct})

    try:
        if not PIPELINE_DEPS_OK:
            q.put({"type": "error", "msg": PIPELINE_IMPORT_ERROR})
            return
        emit("Opening videos…", 2)
        cap_l = cv2.VideoCapture(path_l)
        cap_r = cv2.VideoCapture(path_r)
        if not cap_l.isOpened() or not cap_r.isOpened():
            q.put({"type": "error", "msg": "Could not open video files"}); return

        fps          = int(cap_l.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
        _, f0_l = cap_l.read()
        _, f0_r = cap_r.read()
        emit(f"Frame 0: {f0_l.shape[1]}×{f0_l.shape[0]}  FPS={fps}", 5)

        emit("Detecting blobs…", 8)
        matched_l, matched_r, _ = extract_full_grid(
            find_blobs(f0_l, BLOB_PARAMS), find_blobs(f0_r, BLOB_PARAMS))
        if matched_l is None:
            q.put({"type": "error", "msg": "Grid matching failed on frame 0"}); return
        emit(f"Matched {len(matched_l)} point pairs", 14)

        pts3d_0 = triangulate(matched_l, matched_r)
        faces   = build_mesh(matched_l)
        emit(f"Mesh: {len(faces)} triangles. Starting LK tracking…", 18)

        # ── LK tracking ──────────────────────────────────────────────────────
        cap_l.release(); cap_r.release()
        cap_l = cv2.VideoCapture(path_l)
        cap_r = cv2.VideoCapture(path_r)
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
        timestamps = [0.0]

        fidx = 1
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
            dl = np.linalg.norm((p0_l-pb_l).reshape(-1,2), axis=1)
            dr = np.linalg.norm((p0_r-pb_r).reshape(-1,2), axis=1)
            good = (sl.flatten()==1)&(sr.flatten()==1)&(dl<1.0)&(dr<1.0)

            if good.sum() > 0:
                cur3d[pids[good]] = triangulate(p1_l[good].reshape(-1,2),
                                                p1_r[good].reshape(-1,2))
            all_frames.append(cur3d.flatten().round(3).tolist())
            timestamps.append(round(cap_l.get(cv2.CAP_PROP_POS_MSEC)/1000.0, 6))

            if fidx % 30 == 0:
                pct = 18 + int(77 * fidx / max(total_frames, 1))
                emit(f"Tracking frame {fidx} / {total_frames}  ({good.sum()} pts)", min(pct, 94))

            old_gl=gl.copy(); old_gr=gr.copy()
            p0_l=p1_l[good]; p0_r=p1_r[good]; pids=pids[good]; fidx+=1

        cap_l.release(); cap_r.release()
        emit(f"Tracked {len(all_frames)} frames. Saving…", 96)

        payload = {
            "source": "experiment", "fps": fps,
            "n_points": int(len(matched_l)),
            "faces": faces, "frames": all_frames, "timestamps": timestamps,
            "calib": {"R_ext": R_ext.tolist(), "T_ext": T_ext.flatten().tolist()}
        }
        with open(OUT, "w") as fh:
            json.dump(payload, fh)

        size_mb = os.path.getsize(OUT) / 1e6
        q.put({"type": "done", "frames": len(all_frames),
               "size_mb": round(size_mb, 1), "fps": fps})

    except Exception:
        q.put({"type": "error", "msg": traceback.format_exc()})

# ── Per-request job queue (one job at a time for simplicity) ──────────────────
_job_queue: queue.Queue | None = None
_job_lock = threading.Lock()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.after_request
def disable_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/")
def root():
    return redirect("/render/")


def _normalize_mp4_filename(filename, fallback):
    safe_name = os.path.basename(filename or "") or fallback
    if not safe_name.lower().endswith(".mp4"):
        safe_name = os.path.splitext(safe_name)[0] + ".mp4"
    return safe_name


def _ffmpeg_missing_response():
    return jsonify({"error": "ffmpeg is not installed on the server"}), 500


def _send_mp4_bytes(data, filename):
    data.seek(0)
    return send_file(
        data,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=filename,
    )


def _run_ffmpeg_to_mp4(cmd, dst_path, error_message):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(dst_path):
        return jsonify({
            "error": error_message,
            "stderr": proc.stderr[-4000:],
        }), 500
    with open(dst_path, "rb") as fh:
        return io.BytesIO(fh.read())

@app.route("/render/")
@app.route("/render/index.html")
def serve_render():
    return send_from_directory(RENDER_DIR, "index.html")

@app.route("/render/<path:fname>")
def serve_render_static(fname):
    return send_from_directory(RENDER_DIR, fname)

@app.route("/<path:fname>")
def serve_project(fname):
    return send_from_directory(ROOT, fname)

# ── API: start processing ─────────────────────────────────────────────────────
@app.route("/api/process", methods=["POST"])
def api_process():
    global _job_queue
    if not PIPELINE_DEPS_OK:
        return jsonify({"error": PIPELINE_IMPORT_ERROR}), 500
    for p in (DEFAULT_L, DEFAULT_R):
        if not os.path.isfile(p):
            return jsonify({"error": f"File not found: {p}"}), 400

    with _job_lock:
        if _job_queue is not None:
            return jsonify({"error": "A job is already running"}), 409
        _job_queue = queue.Queue()
        q = _job_queue

    threading.Thread(target=run_pipeline, args=(DEFAULT_L, DEFAULT_R, q), daemon=True).start()
    return jsonify({"ok": True})

# ── API: SSE progress stream ──────────────────────────────────────────────────
@app.route("/api/stream")
def api_stream():
    global _job_queue

    def generate():
        global _job_queue
        q = _job_queue
        if q is None:
            yield "data: {\"type\":\"error\",\"msg\":\"No job running\"}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                with _job_lock:
                    _job_queue = None
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/recording/mp4", methods=["POST"])
def api_recording_mp4():
    upload = request.files.get("recording")
    if upload is None or not upload.filename:
        return jsonify({"error": "Missing uploaded recording"}), 400

    safe_name = _normalize_mp4_filename(
        request.form.get("filename"),
        "compare_recording.mp4",
    )

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return _ffmpeg_missing_response()

    with tempfile.TemporaryDirectory(prefix="hemostasis-recording-") as tmpdir:
        src_name = os.path.basename(upload.filename) or "recording.bin"
        src_path = os.path.join(tmpdir, src_name)
        dst_path = os.path.join(tmpdir, "output.mp4")
        upload.save(src_path)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", src_path,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            dst_path,
        ]
        data = _run_ffmpeg_to_mp4(
            cmd,
            dst_path,
            "ffmpeg failed to convert recording to mp4",
        )
        if not isinstance(data, io.BytesIO):
            return data

    return _send_mp4_bytes(data, safe_name)


@app.route("/api/recording/stitch", methods=["POST"])
def api_recording_stitch():
    left = request.files.get("left")
    right = request.files.get("right")
    if left is None or not left.filename or right is None or not right.filename:
        return jsonify({"error": "Missing left or right uploaded recording"}), 400

    safe_name = _normalize_mp4_filename(
        request.form.get("filename"),
        "compare_recording_both.mp4",
    )

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return _ffmpeg_missing_response()

    with tempfile.TemporaryDirectory(prefix="hemostasis-stitch-") as tmpdir:
        left_path = os.path.join(tmpdir, os.path.basename(left.filename) or "left.bin")
        right_path = os.path.join(tmpdir, os.path.basename(right.filename) or "right.bin")
        dst_path = os.path.join(tmpdir, "stitched.mp4")
        left.save(left_path)
        right.save(right_path)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", left_path,
            "-i", right_path,
            "-filter_complex",
            "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[left];"
            "[1:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[right];"
            "[left][right]hstack=inputs=2,format=yuv420p[v]",
            "-map", "[v]",
            "-an",
            "-c:v", "libx264",
            "-movflags", "+faststart",
            dst_path,
        ]
        data = _run_ffmpeg_to_mp4(
            cmd,
            dst_path,
            "ffmpeg failed to stitch recordings into mp4",
        )
        if not isinstance(data, io.BytesIO):
            return data

    return _send_mp4_bytes(data, safe_name)


def main():
    print("Hemostasis renderer server")
    print(f"Project root: {ROOT}")
    print(f"Output: {OUT}")
    print("Open: http://localhost:8000/render/\n")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
