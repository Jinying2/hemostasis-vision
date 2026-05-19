"""
Convert VisuoShell 3D dot reconstruction output into render experiment JSON.

Usage:
    cd hemostasis
    python render/export_visuoshell_mesh.py
    python render/export_visuoshell_mesh.py --20
    python render/export_visuoshell_mesh.py --15
    python render/export_visuoshell_mesh.py input.csv output.json --fps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISUOSHELL_ROOT = os.path.join(ROOT, "VisuoShell-main")

if VISUOSHELL_ROOT not in sys.path:
    sys.path.insert(0, VISUOSHELL_ROOT)

try:
    from src.loader import load_nodes  # type: ignore
except ModuleNotFoundError:
    load_nodes = None


DEFAULT_INPUT = os.path.join(
    VISUOSHELL_ROOT, "dot_detection_pipline", "example_dot_10mm", "tracked_3d.csv"
)
DEFAULT_OUTPUT = os.path.join(ROOT, "render", "experiment_default.json")
SIZE_PRESETS = {
    "20": {
        "input": os.path.join(ROOT, "20mm", "tracked_3d.csv"),
        "output": os.path.join(ROOT, "render", "experiment_20mm.json"),
    },
    "15": {
        "input": os.path.join(ROOT, "15mm", "tracked_3d.csv"),
        "output": os.path.join(ROOT, "render", "experiment_15mm.json"),
    },
}


def load_nodes_from_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Fallback loader for local tracked_3d.csv files.

    Expected columns: frame, point_id, x, y, z
    Returns a frame-name -> [N, 3] array mapping compatible with VisuoShell's
    loader output.
    """
    rows = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        return {}
    if rows.shape == ():
        rows = np.array([rows], dtype=rows.dtype)

    frames: dict[int, list[tuple[int, float, float, float]]] = {}
    for row in rows:
        frame_id = int(row["frame"])
        point_id = int(row["point_id"])
        frames.setdefault(frame_id, []).append(
            (point_id, float(row["x"]), float(row["y"]), float(row["z"]))
        )

    out: dict[str, np.ndarray] = {}
    for frame_id in sorted(frames):
        pts = sorted(frames[frame_id], key=lambda item: item[0])
        out[f"{frame_id:06d}"] = np.array([[x, y, z] for _, x, y, z in pts], dtype=float)
    return out


def load_nodes_compat(csv_path: str) -> dict[str, np.ndarray]:
    if load_nodes is not None:
        return load_nodes(csv_path)
    return load_nodes_from_csv(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export VisuoShell tracked_3d.csv to render experiment JSON."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--20", dest="size_20", action="store_true", help="Use /20mm/tracked_3d.csv and render/experiment_20mm.json")
    group.add_argument("--15", dest="size_15", action="store_true", help="Use /15mm/tracked_3d.csv and render/experiment_15mm.json")
    return parser.parse_args()


def build_surface_mesh(points: np.ndarray) -> np.ndarray:
    # Project the shell to its best-fit 2D plane before triangulation.
    # ConvexHull in 3D drops concave indentation points, which opens a hole
    # in the middle of the rendered membrane.
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vh[:2].T

    simplices = Delaunay(planar).simplices

    def max_edge_length(tri: np.ndarray) -> float:
        a, b, c = planar[tri[0]], planar[tri[1]], planar[tri[2]]
        return max(
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(a - c),
        )

    edge_lengths = np.array([max_edge_length(tri) for tri in simplices], dtype=float)
    keep = edge_lengths < 3.0 * np.median(edge_lengths)
    return simplices[keep]


def main() -> int:
    args = parse_args()
    preset = None
    if args.size_20:
        preset = SIZE_PRESETS["20"]
    elif args.size_15:
        preset = SIZE_PRESETS["15"]

    input_path = preset["input"] if preset and args.input == DEFAULT_INPUT else args.input
    output_path = preset["output"] if preset and args.output == DEFAULT_OUTPUT else args.output

    nodes_by_time = load_nodes_compat(input_path)
    if not nodes_by_time:
        raise SystemExit(f"No frames found in {input_path}")

    ordered_names = sorted(nodes_by_time)
    first_nodes = nodes_by_time[ordered_names[0]]
    triangles = build_surface_mesh(first_nodes)

    frames = []
    timestamps = []
    for idx, name in enumerate(ordered_names):
        nodes = nodes_by_time[name]
        frames.append(nodes.astype(float).round(6).reshape(-1).tolist())
        timestamps.append(round(idx / args.fps, 6))

    payload = {
        "source": "visuoshell_3d_dot",
        "fps": args.fps,
        "n_points": int(first_nodes.shape[0]),
        "faces": triangles.astype(int).tolist(),
        "frames": frames,
        "timestamps": timestamps,
        "meta": {
            "input": os.path.relpath(os.path.abspath(input_path), ROOT),
            "generator": "render/export_visuoshell_mesh.py",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    size_mb = os.path.getsize(output_path) / 1e6
    print(
        f"Saved {len(frames)} frames, {len(triangles)} faces -> {output_path} ({size_mb:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
