# BioMatrix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18157478.svg)](https://doi.org/10.5281/zenodo.18157478)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/Moz0/biomatrix-demo)

**N-dimensional geometric signal processing engine for drift correction, motion analysis, and topological understanding.**

## Key Features

| Capability | Method | Precision |
|------------|--------|-----------|
| **Drift Detection** | SE(N) Procrustes + residual thresholding | 8.5% relative error |
| **Drift Correction** | Fiducial markers | **0.046 nm** (sub-angstrom) |
| **Motion Analysis** | Velocity + period derivation via Procrustes/FFT | ~5% error |
| **Topology** | Graph degree analysis | Junction detection |
| **Point Cloud Alignment** | Closed-form SE(3) | Real-time capable |

## Technical Specifications

| Property | Value |
|----------|-------|
| **Dimensions** | N-dimensional agnostic (1D, 2D, 3D, ... ND) |
| **Complexity** | O(N·D²) where N=points, D=dimensions |
| **Latency** | ~0.1ms for 1000 pts × 3D |
| **Real-time** | Yes, 30+ fps streaming |

## Installation

```bash
pip install biomatrix
```

Or from source:
```bash
git clone https://github.com/daemind/biomatrix_standalone.git
cd biomatrix_standalone
pip install -e .
```

## Examples

### 1. Drift Correction (Zeiss Plugin)

```python
from biomatrix.plugins.zeiss import LiveTracker, DriftCorrector
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector
import numpy as np

# Real-time tracking
tracker = LiveTracker()
for frame in frames:
    op = tracker.track(frame)
    if op:
        print(f"Drift: {op.t}")

# Sub-nm precision with fiducials
corrector = FiducialDriftCorrector(
    known_fiducial_indices=np.array([100, 101, 102, 103, 104])
)
corrected, drift = corrector.correct(points)
```

### 2. Point Cloud Registration (3D LiDAR)

```python
from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3

# Align two 3D scans
op = derive_procrustes_se3(State(scan1), State(scan2))
if op is not None:
    scan2_aligned = scan2 - op.t  # Apply inverse translation
    print(f"Vehicle motion: {op.t}")
```

### 3. Motion Law Derivation (Walker Dynamics)

```python
# Derive velocity from skeleton sequence via Procrustes
velocities = []
for i in range(1, len(frames)):
    op = derive_procrustes_se3(State(frames[i-1]), State(frames[i]))
    velocities.append(op.t if op else np.zeros(3))

# Derive gait period via FFT
fft = np.abs(np.fft.rfft(limb_positions))
period = 1.0 / np.fft.rfftfreq(len(frames))[np.argmax(fft[1:]) + 1]
```

## Example Scripts

| Script | Description |
|--------|-------------|
| `example_1d_drift.py` | Baseline drift correction on sine signal |
| `example_2d_camera_drift.py` | Star field alignment with outliers |
| `example_audio_drift.py` | Spectral drift removal + FFT validation |
| `demo_lidar_3d.py` | 3D fractal terrain registration |
| `demo_walker_dynamics.py` | Velocity + period derivation from skeleton |
| `demo_stroke_topology.py` | Letter de-occlusion via junction detection |
| `benchmark_zeiss.py` | Scientific validation (6 tests) |

Run examples:
```bash
cd examples
python demo_lidar_3d.py
python demo_walker_dynamics.py
```

## Applications

| Domain | Use Case | Demo |
|--------|----------|------|
| **SMLM Microscopy** | Thermal drift correction | ✓ |
| **Autonomous Vehicles** | LiDAR scan alignment | ✓ |
| **Motion Capture** | Skeleton tracking | ✓ |
| **Robotics** | Joint trajectory analysis | ✓ |
| **Handwriting** | Stroke de-occlusion | ✓ |
| **GPS/Navigation** | Position filtering | - |
| **Audio/Biomedical** | Baseline removal | ✓ |

## Theoretical Basis

**Core Algorithm**: Orthogonal Procrustes (Schönemann, 1966)
- Solves Y = s·R·X + t via SVD decomposition
- Closed-form, no iterations needed
- Optimal in least-squares sense

**Mathematical Structure**: SE(N) Lie Group
- Rigid motions form a group under composition
- Scale + rotation + translation unified

## Assumptions & Limitations

- **Known correspondences**: Point i in frame 1 must correspond to point i in frame 2
- **Rigid motion only**: Does not handle non-rigid deformations
- **Sufficient overlap**: Requires enough shared points between frames
- **Not ICP**: For unknown correspondences, add matching preprocessing

## Interactive Demo

Try the live demo on HuggingFace Spaces:
**https://huggingface.co/spaces/Moz0/biomatrix-demo**

## Citation

```bibtex
@software{biomatrix2026,
  author = {Daemind},
  title = {BioMatrix: N-dimensional Geometric Signal Processing},
  year = {2026},
  doi = {10.5281/zenodo.18157478},
  url = {https://github.com/daemind/biomatrix_standalone}
}
```

## License

MIT License - see [LICENSE](LICENSE)

## References

- Schönemann, P.H. (1966). "A generalized solution of the orthogonal Procrustes problem"
- Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using unit quaternions"
