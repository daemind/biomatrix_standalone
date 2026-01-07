# BioMatrix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18157478.svg)](https://doi.org/10.5281/zenodo.18157478)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/Moz0/biomatrix-demo)

**Algebraic transformation inference engine for N-dimensional geometric signal processing.**

BioMatrix automatically derives transformations between point sets using pure algebraic methods—no neural networks, no training data, no iterative optimization.

## Core Innovation

Given two point sets (Input, Output), BioMatrix **algebraically derives** the transformation T such that:

```
T(Input) = Output
```

This is **not** pattern matching or machine learning. It is **mathematical inference**:

| Capability | Method | What it Derives |
|------------|--------|-----------------|
| **Isometry Detection** | Orthogonal Procrustes | Rotation R, Translation t, Scale s |
| **Subset Selection** | Set Algebra | Which points were kept/deleted |
| **Causality Separation** | Intersection/Difference | Inert vs Causal points |
| **Symmetry Detection** | Group Theory (D₄, Bₙ) | Reflections, rotations, tilings |
| **Lifting** | Kernel Linearization | Feature space transformations |
| **Multi-component** | Union Decomposition | Per-object transforms |

---

## Mathematical Foundation

### 1. State Space: ℝⁿ Point Clouds

A **State** is a finite subset S ⊂ ℝⁿ. All operations are dimension-agnostic:

```python
from biomatrix.core.state import State

# 2D, 3D, or any dimension
s = State(np.array([[0,0], [1,0], [0,1], [1,1]]))  # 2D square
s = State(np.random.randn(100, 10))  # 100 points in 10D
```

### 2. Operator Monoïd

Transformations form a **monoïd** under composition:
- **Identity**: I(S) = S
- **Affine**: f(x) = xAᵀ + b
- **Selection**: σᵥ(S) = {x ∈ S | x[d] = v}
- **Union**: (T₁ ∪ T₂)(S) = T₁(S₁) ∪ T₂(S₂)
- **Sequential**: (T₁ ∘ T₂)(S) = T₁(T₂(S))

```python
from biomatrix.core.transform import AffineTransform

# Compose transformations
T1 = AffineTransform.translate([1, 0])
T2 = AffineTransform.rotation_plane(np.pi/2, 0, 1)
T = T1.compose(T2)  # Rotation then translation
```

### 3. Derivation Pipeline

`derive_transformation(s_in, s_out)` attempts derivation in order:

1. **Identity**: s_in == s_out?
2. **Subset**: s_out ⊂ s_in (deletion/selection)
3. **Affine**: Procrustes (rotation + translation + scale)
4. **Lifting**: Kernel space linearization (tiling, fractals)
5. **Causality**: Separate inert (intersection) from causal (difference)
6. **Union**: Multi-component decomposition

---

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

---

## API Reference

### Core: Transform Derivation

```python
from biomatrix.core.state import State
from biomatrix.core.derive import derive_transformation

# Automatic law discovery
s_in = State(input_points)
s_out = State(output_points)

T = derive_transformation(s_in, s_out)
if T:
    result = T.apply(s_in)  # Should equal s_out
```

### AffineTransform

```python
from biomatrix.core.transform import AffineTransform

# Factory methods
T = AffineTransform.identity(n_dims=3)
T = AffineTransform.translate([dx, dy, dz])
T = AffineTransform.scale(factor=2.0, n_dims=3)
T = AffineTransform.rotation_plane(angle=np.pi/4, dim1=0, dim2=1, n_dims=3)
T = AffineTransform.reflection(normal=[1, 0, 0])

# Decomposition
components = T.decompose()  # Returns rotation, scale, translation
```

### Procrustes (Explicit)

```python
from biomatrix.core.derive.procrustes import derive_procrustes_se3

op = derive_procrustes_se3(s_from, s_to)
# Returns ProcrustesOperator with:
#   op.R  - Rotation matrix (D×D)
#   op.t  - Translation vector (D,)
#   op.s  - Scale factor
```

### Causality Detection

```python
from biomatrix.core.derive.algebra import derive_causality

# Separate static from dynamic points
op = derive_causality(s_before, s_after)
# Returns UnionOperator([InertOp, CausalOp])
```

### Topology

```python
from biomatrix.core.topology import (
    partition_by_connectivity,  # Connected components
    partition_by_value,         # Group by dimension value
    get_extremities,            # Endpoints (degree=1)
    get_euler_number,           # χ = V - E
    is_hollow,                  # Has internal holes?
)

objects = partition_by_connectivity(state)  # List[State]
```

---

## Example Demos

### Automatic Transformation Derivation

```bash
python examples/demo_transform_derivation.py
```

Demonstrates automatic derivation of:
- Translation
- Rotation
- Scaling
- Reflection
- Subset selection

**Output**: 5/5 tests pass

### Causality Detection

```bash
python examples/demo_causality.py
```

Separates inert (unchanged) from causal (changed) points between two states.

### Drift Correction (Zeiss Plugin)

```python
from biomatrix.plugins.zeiss import LiveTracker, DriftCorrector
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector

# Real-time streaming
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

---

## Technical Specifications

| Property | Value |
|----------|-------|
| **Dimensions** | N-dimensional agnostic (1D, 2D, 3D, ... ND) |
| **Complexity** | O(N·D²) for Procrustes |
| **Latency** | ~0.1ms for 1000 pts × 3D |
| **Real-time** | Yes, 30+ fps streaming |
| **Precision** | 0.046 nm (sub-angstrom with fiducials) |

---

## Module Structure

```
biomatrix/
├── core/
│   ├── state.py          # State class (point cloud container)
│   ├── transform.py      # AffineTransform with compositions
│   ├── topology.py       # Connected components, Euler, extremities
│   ├── signatures.py     # Geometric signatures for matching
│   ├── query.py          # Query-based selection
│   ├── derive/
│   │   ├── core.py       # derive_transformation() master
│   │   ├── algebra.py    # Lifting, causality, tiling
│   │   ├── affine.py     # Procrustes variants
│   │   ├── permutation.py# Value/component permutations
│   │   ├── union.py      # Multi-component transforms
│   │   └── procrustes.py # SE(N) Procrustes
│   └── operators/
│       ├── core.py       # Union, Repeat, Sequence
│       ├── lifting.py    # Lift/Slice operators
│       ├── selection.py  # Query-based selection
│       └── algebra.py    # Composition, Folding
└── plugins/
    └── zeiss/            # SMLM microscopy drift correction
```

---

## Applications

| Domain | Use Case | Precision |
|--------|----------|-----------|
| **SMLM Microscopy** | Thermal drift correction | 0.046 nm |
| **Autonomous Vehicles** | LiDAR scan alignment | Real-time |
| **Motion Capture** | Skeleton tracking | ~5% velocity error |
| **Robotics** | Joint trajectory analysis | - |
| **Handwriting** | Stroke de-occlusion | Junction detection |
| **Fraud Detection** | Transaction anomaly | Mahalanobis distance |
| **Sensor Fusion** | IMU+UWB positioning | 1.6x improvement |

---

## Theoretical Basis

### Orthogonal Procrustes (Schönemann, 1966)

Solves min‖Y - s·R·X - t‖² via SVD:

```
H = (X - μₓ)ᵀ(Y - μᵧ)
U, S, Vᵀ = svd(H)
R = V·Uᵀ
t = μᵧ - s·R·μₓ
```

### SE(N) Lie Group

Rigid motions form a group:
- Closure: T₁ ∘ T₂ is a rigid motion
- Identity: I(x) = x
- Inverse: T⁻¹ exists

### Lifting Kernels

Linearize nonlinear transformations by embedding:

```
X ∈ ℝᴺˣᴰ → φ(X) ∈ ℝᴺˣ⁽ᴰ⁺ᴸ⁾
```

Kernels: Identity, Kronecker, Topological, Distance-rank, Poly2, Symmetry

---

## Assumptions & Limitations

- **Known correspondences**: Point i in frame 1 must correspond to point i in frame 2 (for Procrustes)
- **Rigid motion only**: Does not handle non-rigid deformations
- **Not ICP**: For unknown correspondences, add matching preprocessing

---

## Citation

```bibtex
@software{biomatrix2026,
  author = {Daemind},
  title = {BioMatrix: Algebraic Transformation Inference Engine},
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
- Gower, J.C. (1975). "Generalized Procrustes analysis"
