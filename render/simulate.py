"""Generate a spring-based simulation from the experimental mesh JSON.

Usage:
    python render/simulate.py
    python render/simulate.py --20
    python render/simulate.py --15
"""

from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "viewer" / "stereo_experiment.json"
OUT_PATH = ROOT / "viewer" / "stereo_simulation.json"
SIZE_PRESETS = {
    "20": {
        "input": ROOT / "render" / "experiment_20mm.json",
        "output": ROOT / "render" / "simulation_20mm.json",
    },
    "15": {
        "input": ROOT / "render" / "experiment_15mm.json",
        "output": ROOT / "render" / "simulation_15mm.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spring-based simulation from an experiment mesh JSON."
    )
    parser.add_argument("input", nargs="?", default=str(IN_PATH))
    parser.add_argument("output", nargs="?", default=str(OUT_PATH))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--20", dest="size_20", action="store_true", help="Use render/experiment_20mm.json -> render/simulation_20mm.json")
    group.add_argument("--15", dest="size_15", action="store_true", help="Use render/experiment_15mm.json -> render/simulation_15mm.json")
    return parser.parse_args()


def smooth_edge_pad(seq, smooth_win):
    """Moving average with edge-padding to avoid boundary distortion."""
    half_w = smooth_win // 2
    padded = np.concatenate(
        [seq[:1].repeat(half_w, axis=0), seq, seq[-1:].repeat(half_w, axis=0)],
        axis=0,
    )
    return np.stack(
        [padded[t:t + smooth_win].mean(axis=0) for t in range(len(seq))],
        axis=0,
    )

def resolve_paths(args):
    if args.size_20:
        preset = SIZE_PRESETS["20"]
    elif args.size_15:
        preset = SIZE_PRESETS["15"]
    else:
        preset = None

    in_path = Path(preset["input"] if preset and args.input == str(IN_PATH) else args.input)
    out_path = Path(preset["output"] if preset and args.output == str(OUT_PATH) else args.output)
    return in_path, out_path


def main():
    args = parse_args()
    in_path, out_path = resolve_paths(args)

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    faces = data["faces"]
    fps = data.get("fps", 30)
    exp_frames = [np.array(frame).reshape(-1, 3) for frame in data["frames"]]
    X0 = exp_frames[0]
    num_vertices = len(X0)
    num_frames = len(exp_frames)

    print(f"Loaded mesh: {in_path}")
    print(f"  {num_vertices} vertices, {len(faces)} faces, {num_frames} exp frames")

    edge_count = {}
    for tri in faces:
        i, j, k = tri
        for a, b in ((i, j), (j, k), (k, i)):
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1

    edges = list(edge_count.keys())
    print(f"Edges: {len(edges)}")

    rest_len = {(i, j): np.linalg.norm(X0[i] - X0[j]) for (i, j) in edges}

    boundary_nodes = set()
    for (i, j), cnt in edge_count.items():
        if cnt == 1:
            boundary_nodes.add(i)
            boundary_nodes.add(j)
    print(f"Boundary nodes: {len(boundary_nodes)}")

    interior_mask = np.array([i not in boundary_nodes for i in range(num_vertices)])
    all_disps = np.stack(exp_frames, axis=0) - X0
    node_max_disp = np.linalg.norm(all_disps, axis=2).max(axis=0)
    drive_weights_all = (
        node_max_disp / node_max_disp.max() if node_max_disp.max() > 0 else np.zeros(num_vertices)
    )

    driven_nodes = [i for i in range(num_vertices) if interior_mask[i]]
    driven_arr = np.array(driven_nodes)
    driven_weights = drive_weights_all[driven_arr] ** 0.4
    print(
        f"Driven region (all interior): {len(driven_nodes)} nodes  "
        f"(weight range [{driven_weights.min():.3f}, {driven_weights.max():.2f}])"
    )

    smooth_win = 7
    exp_driven_raw = np.stack([frame[driven_arr] for frame in exp_frames], axis=0)
    exp_driven = np.stack(
        [smooth_edge_pad(exp_driven_raw[:, m, :], smooth_win) for m in range(exp_driven_raw.shape[1])],
        axis=1,
    )
    max_disp = np.linalg.norm(exp_driven - X0[driven_arr], axis=2).max()
    print(f"Per-node drive range: {max_disp:.3f} mm max  (smoothed win={smooth_win}, edge-padded)")

    k_s = 135.0
    k_restore = 2.0
    k_boundary = 2.0
    c_press = 1.5
    c_release = 3.0
    frame_drive_eps = 0.15
    mass = 1.0
    alpha = 0.45
    drive_scale = 1.30
    dt = 1.0 / fps

    print(f"dt={dt:.4f}s  (1:1 with experiment frames)")

    X = X0.copy()
    V = np.zeros_like(X)
    out_frames = [X.flatten().tolist()]
    contact_started = False
    prev_frame_mag = 0.0
    beta = 0.2
    core_disp_smooth = np.zeros_like(exp_driven[0])

    for t_idx in range(1, num_frames):
        # 8.1 Spring forces
        F = np.zeros_like(X)
        for (i, j) in edges:
            dv = X[i] - X[j]
            length = np.linalg.norm(dv)
            if length < 1e-8:
                continue
            f = -k_s * (length - rest_len[(i, j)]) * (dv / length)
            F[i] += f
            F[j] -= f

        # 8.2 Global restoring force: gently pulls mesh back to rest shape
        F -= k_restore * (X - X0)

        # 8.2b Soft boundary anchor (replaces hard lock)
        for i in boundary_nodes:
            F[i] -= k_boundary * (X[i] - X0[i])

        # 8.3 Phase-aware damping: higher during release to suppress overshoot
        core_disps_now = exp_driven[t_idx] - X0[driven_arr]
        core_disp_smooth = beta * core_disps_now + (1 - beta) * core_disp_smooth
        core_mean = core_disp_smooth.mean(axis=0)
        frame_mag_now = np.linalg.norm(core_disp_smooth, axis=1).mean()
        c = c_release if (contact_started and frame_mag_now < prev_frame_mag) else c_press
        prev_frame_mag = frame_mag_now
        F -= c * V

        # 8.4 Euler integration
        V_new = V + dt * (F / mass)
        X_new = X + dt * V_new

        # 8.5 Drive core nodes toward per-node experimental target.
        if frame_mag_now >= frame_drive_eps:
            contact_started = True
            for k, i in enumerate(driven_nodes):
                weight = driven_weights[k]
                blended_disp = 0.10 * core_mean + 0.90 * core_disp_smooth[k]
                target = X0[i] + weight * drive_scale * blended_disp
                X_new[i] = (1 - alpha) * X_new[i] + alpha * target
                V_new[i] *= (1 - alpha)
        elif not contact_started:
            for i in range(num_vertices):
                if i not in boundary_nodes:
                    X_new[i] = X0[i]
                    V_new[i] = 0.0

        X, V = X_new, V_new
        out_frames.append(X.flatten().tolist())

    print(f"Simulation done. Frames saved: {len(out_frames)}")

    out = {
        "frames": out_frames,
        "faces": faces,
        "fps": fps,
        "source": "simulation",
        "meta": {
            "input": str(in_path.relative_to(ROOT) if in_path.is_relative_to(ROOT) else in_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
