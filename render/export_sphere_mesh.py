"""
Convert sphere_3D_nodes.csv (VisuoShell sphere reconstruction output) to
render/sphere_3D_mesh_NNmm.json for use in sphere_preview.html and compare_real_video.html.

The CSV has columns: frame, point_id, x, y, z
Points are arranged in longitude strips: point_id = strip_index * pts_per_strip + row.
All row-0 points share the same position (the pole).

Usage:
    cd hemostasis
    python render/export_sphere_mesh.py --20   # uses 20mm/sphere_3D_nodes.csv
    python render/export_sphere_mesh.py --15   # uses 15mm/sphere_3D_nodes.csv
    python render/export_sphere_mesh.py input.csv output.json --fps 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIZE_PRESETS = {
    "20": {
        "input": os.path.join(ROOT, "20mm", "sphere_3D_nodes.csv"),
        "output": os.path.join(ROOT, "render", "sphere_3D_mesh_20.json"),
    },
    "15": {
        "input": os.path.join(ROOT, "15mm", "sphere_3D_nodes.csv"),
        "output": os.path.join(ROOT, "render", "sphere_3D_mesh_15.json"),
    },
}

DEFAULT_INPUT = SIZE_PRESETS["20"]["input"]
DEFAULT_OUTPUT = os.path.join(ROOT, "render", "sphere_default.json")


def load_csv(csv_path: str) -> dict[int, list[list[float]]]:
    """Read sphere_3D_nodes.csv -> {frame_id: [[x,y,z], ...]} sorted by point_id."""
    import csv
    frames: dict[int, list[tuple[int, float, float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = int(row["frame"])
            pid = int(row["point_id"])
            frames.setdefault(fid, []).append(
                (pid, float(row["x"]), float(row["y"]), float(row["z"]))
            )
    out: dict[int, list[list[float]]] = {}
    for fid in sorted(frames):
        pts = sorted(frames[fid], key=lambda r: r[0])
        out[fid] = [[x, y, z] for _, x, y, z in pts]
    return out


def detect_strip_params(pts: list[list[float]]) -> tuple[int, int]:
    """Return (n_strips, pts_per_strip) by finding the pole repetition period.

    The pole point (row 0 of each strip) coincides with point 0. We scan
    forward from index 1 until we find a point whose x,z match point 0.
    """
    p0 = pts[0]
    for i in range(1, len(pts)):
        p = pts[i]
        if abs(p[0] - p0[0]) < 0.01 and abs(p[2] - p0[2]) < 0.01:
            pts_per_strip = i
            n_strips = len(pts) // pts_per_strip
            if n_strips * pts_per_strip == len(pts):
                return n_strips, pts_per_strip
    raise ValueError(
        f"Could not detect strip period from {len(pts)} points. "
        "Expected pole-repeat pattern."
    )


def build_faces(n_strips: int, pts_per_strip: int) -> list[list[int]]:
    """Build triangle faces for the strip mesh (pole cap excluded — JS adds it).

    For each strip si and row j in [1, pts_per_strip-2], two triangles fill
    the quad between strip si and strip (si+1)%n_strips.
    """
    faces = []
    nS, pS = n_strips, pts_per_strip
    for si in range(nS):
        si_next = (si + 1) % nS
        for j in range(1, pS - 1):
            a = si * pS + j
            b = si * pS + j + 1
            c = si_next * pS + j + 1
            d = si_next * pS + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return faces


def compute_center_radius(pts: list[list[float]]) -> tuple[list[float], float]:
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    r = sum(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) for p in pts) / n
    return [round(cx, 4), round(cy, 4), round(cz, 4)], round(r, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sphere_3D_nodes.csv to render sphere JSON."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--20", dest="size_20", action="store_true",
                       help="Use 20mm/sphere_3D_nodes.csv -> render/sphere_3D_mesh_20.json")
    group.add_argument("--15", dest="size_15", action="store_true",
                       help="Use 15mm/sphere_3D_nodes.csv -> render/sphere_3D_mesh_15.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.size_20:
        preset = SIZE_PRESETS["20"]
    elif args.size_15:
        preset = SIZE_PRESETS["15"]
    else:
        preset = None

    input_path = preset["input"] if preset and args.input == DEFAULT_INPUT else args.input
    output_path = preset["output"] if preset and args.output == DEFAULT_OUTPUT else args.output

    if not os.path.exists(input_path):
        raise SystemExit(f"Input not found: {input_path}")

    print(f"Loading {input_path} …")
    frames_data = load_csv(input_path)
    if not frames_data:
        raise SystemExit(f"No frames found in {input_path}")

    ordered_ids = sorted(frames_data)
    first_pts = frames_data[ordered_ids[0]]

    n_strips, pts_per_strip = detect_strip_params(first_pts)
    print(f"Detected {n_strips} strips × {pts_per_strip} pts/strip = {n_strips * pts_per_strip} points")

    faces = build_faces(n_strips, pts_per_strip)
    center, radius = compute_center_radius(first_pts)

    frame_list = []
    timestamps = []
    for idx, fid in enumerate(ordered_ids):
        pts = frames_data[fid]
        flat = [round(v, 6) for p in pts for v in p]
        frame_list.append(flat)
        timestamps.append(round(idx / args.fps, 6))

    payload = {
        "source": "sphere_3D_nodes",
        "fps": args.fps,
        "n_points": len(first_pts),
        "n_strips": n_strips,
        "pts_per_strip": pts_per_strip,
        "faces": faces,
        "center": center,
        "radius": radius,
        "frames": frame_list,
        "timestamps": timestamps,
        "meta": {
            "input": os.path.relpath(os.path.abspath(input_path), ROOT),
            "generator": "render/export_sphere_mesh.py",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    size_mb = os.path.getsize(output_path) / 1e6
    print(
        f"Saved {len(frame_list)} frames, {len(faces)} faces "
        f"-> {output_path} ({size_mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
