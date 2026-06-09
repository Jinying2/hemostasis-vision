# Hemostasis — Vision Subsystem

Stereo fisheye camera pipeline for tracking dot-grid deformation on a soft membrane, used in a robotic ultrasound-guided hemostasis system.

## Overview

Two monocular fisheye cameras observe a 15×15 dot-grid membrane (~225 markers). The pipeline detects, matches, tracks, and triangulates the dots to produce 3D surface deformation data.

```
Blob detection → Stereo matching → LK optical flow → Triangulation → 3D mesh
```

## Demo

### 3D Mesh Viewer

<video src="demo/viewer.mov" controls width="100%"></video>

Interactive Three.js viewer showing the triangulated 3D surface of the dot-grid membrane across frames.

### Experiment vs Simulation Comparison

<video src="demo/compare.mov" controls width="100%"></video>

Side-by-side view of the tracked experimental mesh against the spring-mass simulation (`render/compare.html`).

### Experiment vs Real Video

<video src="demo/real_video_viewer.mp4" controls width="100%"></video>

Side-by-side view of the experimental mesh against the real fisheye video feed (`render/compare_real_video.html`).

## Structure

```
hemostasis/
├── full-blob-find.py          # Main pipeline script (blob detection + LK tracking + 3D viz)
├── utils.py                   # Shared pipeline functions (blob detect, grid extract, plotting)
├── fisheye_stereo_calib.npz   # Stereo calibration (K, D, R, T for both cameras)
│
├── 15mm/ 20mm/                # Experiment data per indentation depth
│   ├── raw_left_*.mp4         # Raw stereo video
│   ├── tracked_2d_left/right.csv
│   ├── tracked_3d.csv
│   └── sphere_3D_nodes.csv    # Sphere surface reconstruction (per frame)
│
├── render/
│   ├── index.html             # Three.js mesh viewer
│   ├── compare.html           # Side-by-side experiment vs simulation
│   ├── compare_real_video.html
│   ├── server.py              # Flask dev server (runs pipeline + serves viewer)
│   ├── export_visuoshell_mesh.py  # Convert tracked_3d.csv → mesh JSON
│   ├── export_sphere_mesh.py      # Convert sphere_3D_nodes.csv → sphere JSON
│   └── simulate.py            # Spring-mass simulation from experimental mesh
│
└── viewer/
    ├── index.html             # Standalone Three.js viewer
    └── run_and_export.py      # Run pipeline and write stereo_experiment.json
```

## Setup

```bash
pip install opencv-python numpy scipy matplotlib flask
```

`render/export_visuoshell_mesh.py` can read the local `15mm/tracked_3d.csv` and `20mm/tracked_3d.csv` files directly. A `VisuoShell-main/` checkout is optional.

## Usage

### 1. Run the main pipeline (live visualization)

```bash
cd hemostasis
python full-blob-find.py
```

Videos are hardcoded at the top of the file — edit `VIDEO_L` / `VIDEO_R` to point to your stereo pair.

### 2. Export mesh for the 3D viewer

```bash
python viewer/run_and_export.py
# or with custom videos:
python viewer/run_and_export.py left.mp4 right.mp4
```

Then serve and open the viewer:

```bash
python -m http.server 8000
# open http://localhost:8000/viewer/
```

### 3. Flask dev server (pipeline + viewer in one)

```bash
python render/server.py
# open http://localhost:8000/render/
```

### 4. Export VisuoShell tracked data to mesh JSON

```bash
python render/export_visuoshell_mesh.py --20   # uses 20mm/tracked_3d.csv
python render/export_visuoshell_mesh.py --15   # uses 15mm/tracked_3d.csv
```

### 5. Export sphere reconstruction to mesh JSON

```bash
python render/export_sphere_mesh.py --20   # uses 20mm/sphere_3D_nodes.csv
python render/export_sphere_mesh.py --15   # uses 15mm/sphere_3D_nodes.csv
```

### 6. Run spring-mass simulation

```bash
python render/simulate.py --20
python render/simulate.py --15
```

## Calibration

`fisheye_stereo_calib.npz` stores `K_l, D_l, K_r, D_r` (fisheye intrinsics) and `R_ext, T_ext` (extrinsics). The rotation/translation values are also hardcoded inline in each script as a fallback.

## Known Issues

- LK tracking drifts on fast dot motion. Auto-recovery triggers when tracked count drops below 225, but the 15% area tolerance is strict.
