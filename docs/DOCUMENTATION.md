# BioMatrix Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Concepts](#core-concepts)
3. [State API](#state-api)
4. [Transform API](#transform-api)
5. [Derivation Pipeline](#derivation-pipeline)
6. [Operators](#operators)
7. [Topology Module](#topology-module)
8. [Zeiss Plugin](#zeiss-plugin)
9. [Mathematical Theory](#mathematical-theory)
10. [Performance](#performance)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      BioMatrix Engine                       │
├─────────────────────────────────────────────────────────────┤
│  derive_transformation(S_in, S_out) → Operator              │
│    ├── Identity Check                                        │
│    ├── Subset Detection (Selection, Deletion)               │
│    ├── Affine Derivation (Procrustes, Matched, Centered)    │
│    ├── Lifting (Kernel Linearization)                       │
│    ├── Causality (Inert vs Causal Separation)               │
│    └── Union (Multi-component Decomposition)                 │
├─────────────────────────────────────────────────────────────┤
│  Operator.apply(State) → State                              │
│    ├── AffineTransform (f(x) = xA^T + b)                    │
│    ├── SelectionOperator (σ_v)                               │
│    ├── UnionOperator (T₁ ∪ T₂)                              │
│    ├── SequentialOperator (T₁ ∘ T₂)                         │
│    └── LiftingOperator (φ: ℝᴰ → ℝᴰ⁺ᴸ)                       │
├─────────────────────────────────────────────────────────────┤
│  State: S ⊂ ℝⁿ (N-dimensional point cloud)                  │
│    ├── points: np.ndarray (N, D)                            │
│    ├── centroid, spread, bbox_min, bbox_max                 │
│    └── Set operations: union, intersection, difference      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Dimension Agnosticism

BioMatrix treats all dimensions uniformly. There is no special handling for x, y, z, color, time, etc. Every coordinate is just a dimension in ℝⁿ.

```python
# 2D points
State(np.array([[0, 0], [1, 0], [0, 1]]))

# 3D points
State(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))

# N-D points (e.g., transaction features)
State(np.array([[t, amount, loc_x, loc_y, device, channel] for ...]))
```

### Algebraic Operators

Transformations are **operators** that map State → State:

```python
T: State → State
T.apply(s_in) → s_out
```

Operators form a **monoïd**:
- **Closure**: T₁ ∘ T₂ is an operator
- **Associativity**: (T₁ ∘ T₂) ∘ T₃ = T₁ ∘ (T₂ ∘ T₃)
- **Identity**: I ∘ T = T ∘ I = T

### Derivation vs Fitting

BioMatrix **derives** transformations, it does not **fit** them:

| Fitting (ML) | Derivation (Algebraic) |
|--------------|----------------------|
| Requires training data | Works on single example |
| Minimizes empirical risk | Finds exact solution |
| Generalizes statistically | Generalizes mathematically |
| Iterative optimization | Closed-form solution |

---

## State API

### Creation

```python
from biomatrix.core.state import State

# From numpy array
s = State(points)  # points: (N, D) array

# From grid (embedded as N+1 dimensional)
s = State.from_grid(grid)  # grid[x, y] → (x, y, value)
```

### Properties

```python
s.n_points   # int: number of points
s.n_dims     # int: dimensionality
s.centroid   # np.ndarray (D,): mean of points
s.spread     # np.ndarray (D,): max - min per dimension
s.bbox_min   # np.ndarray (D,): minimum coordinates
s.bbox_max   # np.ndarray (D,): maximum coordinates
s.is_empty   # bool: n_points == 0
```

### Set Operations

```python
s1.union(s2)         # S₁ ∪ S₂
s1.intersection(s2)  # S₁ ∩ S₂
s1.difference(s2)    # S₁ \ S₂
s1.is_subset_of(s2)  # S₁ ⊆ S₂
s1 == s2             # Set equality
```

### Mathematical Morphology

```python
s.dilation(structuring_element)   # Minkowski sum: A ⊕ B
s.erosion(structuring_element)    # Minkowski difference: A ⊖ B
```

---

## Transform API

### AffineTransform

```python
from biomatrix.core.transform import AffineTransform

# Affine transformation: f(x) = xA^T + b
T = AffineTransform(linear=A, translation=b)
```

### Factory Methods

```python
# Identity
T = AffineTransform.identity(n_dims=3)

# Translation
T = AffineTransform.translate([dx, dy, dz])

# Uniform scaling
T = AffineTransform.scale(factor=2.0, n_dims=3)

# Non-uniform scaling
T = AffineTransform.scale(factors=[sx, sy, sz])

# Rotation in plane
T = AffineTransform.rotation_plane(angle, dim1=0, dim2=1, center=None, n_dims=3)

# Rotation from matrix
T = AffineTransform.rotation_from_matrix(R, center=None)

# Reflection across hyperplane
T = AffineTransform.reflection(normal=[1, 0, 0], origin=None)

# Projection onto hyperplane
T = AffineTransform.projection(dim=2, value=0, n_dims=3)
```

### Composition

```python
T1 = AffineTransform.translate([1, 0])
T2 = AffineTransform.rotation_plane(np.pi/2, 0, 1)

# Compose: T1 ∘ T2 (apply T2 first, then T1)
T = T1.compose(T2)

# Or using @
T = T1 @ T2
```

### Decomposition

```python
components = T.decompose()
# Returns dict with:
#   'rotation': Orthogonal matrix R
#   'scale': Symmetric matrix U (from polar decomposition)
#   'translation': Vector b
#   'determinant': det(R) (+1 for rotation, -1 for reflection)
#   'scale_factors': Per-axis scaling (eigenvalues of U)
```

---

## Derivation Pipeline

### Master Resolver

```python
from biomatrix.core.derive import derive_transformation

T = derive_transformation(s_in, s_out)
```

The pipeline attempts derivation in order:

1. **Identity**: `s_in == s_out` → IdentityOperator
2. **Subset**: `s_out ⊂ s_in` → derive_subset()
3. **Matched Affine**: Same cardinality → derive_matched_affine()
4. **Lifting**: Kernel linearization → derive_lift_and_slice()
5. **Causality**: Inert/Causal separation → derive_causality()
6. **Union**: Multi-component → derive_union()

### Individual Derivers

```python
from biomatrix.core.derive import (
    derive_transformation,     # Master resolver
    derive_subset,             # Selection/Deletion
    derive_matched_affine,     # Procrustes with correspondences
    derive_affine_centered,    # Centered alignment
    derive_translation_robust, # Robust translation
    derive_value_permutation,  # Dimension value swap
    derive_component_permutation,  # Object reordering
)

from biomatrix.core.derive.algebra import (
    derive_causality,          # Inert vs Causal
    derive_lift_and_slice,     # Kernel lifting
    derive_tiling_lift_strategy,   # Tiling detection
    derive_homothety_lift_strategy,# Zoom/resample
)

from biomatrix.core.derive.procrustes import (
    derive_procrustes_se3,     # SE(N) closed-form
    ProcrustesOperator,        # Result container
)
```

---

## Operators

### Basic Operators

```python
from biomatrix.core.base import Operator, IdentityOperator, SequentialOperator

# Identity
I = IdentityOperator()

# Sequential composition
T = SequentialOperator([T1, T2, T3])  # T3 ∘ T2 ∘ T1
```

### Core Operators

```python
from biomatrix.core.operators import (
    UnionOperator,            # Apply different transforms to different components
    RepeatOperator,           # Tile/repeat pattern
    LinearSequenceOperator,   # Arithmetic sequence
    CropToComponentOperator,  # Select single component
    SelectThenActOperator,    # Selection + transformation
)
```

### Affine Operators

```python
from biomatrix.core.operators import (
    TranslationOperator,
    RotationOperator,
    ReflectionOperator,
    ScaleOperator,
)
```

### Selection Operators

```python
from biomatrix.core.operators import (
    SelectBySignatureOperator,  # Filter by geometric signature
    PartitionOperator,          # Split into components
)
```

---

## Topology Module

### Connected Components

```python
from biomatrix.core.topology import (
    partition_by_connectivity,  # List[State] of connected components
    partition_by_value,         # Group by dimension value
    get_component_labels,       # (N,) int array of component IDs
)

objects = partition_by_connectivity(state, mode='moore')
# mode: 'moore' (8-connected), 'von_neumann' (4-connected), 'knn', 'adaptive'
```

### Topological Invariants

```python
get_euler_number(state)  # χ = V - E (or objects - holes for grids)
is_hollow(state)         # Has internal cavities?
is_convex(state)         # Is convex hull == object?
get_extremities(state)   # Points with degree 1
get_topology_vector(state)  # Feature vector for matching
```

### Adjacency

```python
from biomatrix.core.topology import get_adjacency_matrix

A = get_adjacency_matrix(state, mode='moore')
# Returns scipy.sparse.csr_matrix
```

---

## Zeiss Plugin

### Real-time Drift Tracking

```python
from biomatrix.plugins.zeiss import LiveTracker

tracker = LiveTracker()
for frame in stream:
    op = tracker.track(frame)  # Returns ProcrustesOperator or None
    if op:
        print(f"Drift: {op.t}")
```

### Cumulative Drift Correction

```python
from biomatrix.plugins.zeiss import DriftCorrector

corrector = DriftCorrector()
for frame in frames:
    corrected = corrector.correct(frame)
```

### Sub-nm Fiducial Correction

```python
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector

corrector = FiducialDriftCorrector(
    known_fiducial_indices=np.array([100, 101, 102, 103, 104])
)
corrected, drift = corrector.correct(points)
# Precision: 0.046 nm
```

### Fiber Tracking

```python
from biomatrix.plugins.zeiss import FiberLinker

linker = FiberLinker(min_length=5)
for frame_idx, frame in enumerate(frames):
    linker.update(frame, frame_idx)

tracks = linker.get_tracks()  # Dict[int, List[np.ndarray]]
```

---

## Mathematical Theory

### Orthogonal Procrustes Problem

Given X, Y ∈ ℝⁿˣᵈ, find R ∈ O(d), s > 0, t ∈ ℝᵈ minimizing:

```
min ‖Y - s·X·R - 1·tᵀ‖²_F
```

**Closed-form solution** (Schönemann 1966):

```
μₓ = mean(X), μᵧ = mean(Y)
X̃ = X - μₓ, Ỹ = Y - μᵧ
H = X̃ᵀỸ
U, Σ, Vᵀ = svd(H)
R = V·Uᵀ
s = trace(Σ) / ‖X̃‖²_F
t = μᵧ - s·R·μₓ
```

### SE(N) Lie Group

The Special Euclidean group SE(n) = SO(n) ⋉ ℝⁿ:
- SO(n): Rotations (det R = +1)
- ℝⁿ: Translations

BioMatrix uses SE(n) ∪ O(n) to include reflections (det R = ±1).

### Lifting Kernels

Embed ℝᴰ → ℝᴰ⁺ᴸ to linearize nonlinear relations:

| Kernel | Embedding | Use Case |
|--------|-----------|----------|
| Identity | X → X | Affine transforms |
| Topological | X → [X, connectivity] | Folding, boundary |
| Distance-rank | X → [X, ‖x - centroid‖] | Radial ordering |
| Kronecker | X → [X, X ⊗ basis] | Tiling, fractals |
| Poly2 | X → [X, X², XY] | Quadratic relations |

---

## Performance

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Procrustes | O(N·D²) | O(D²) |
| Adjacency (Moore) | O(N²) | O(N²) |
| Connected Components | O(N·α(N)) | O(N) |
| Lifting | O(N·D·L) | O(N·(D+L)) |

### Benchmarks

| Test | Result |
|------|--------|
| Drift Detection | 8.5% relative error |
| Drift Correction (fiducial) | 0.046 nm precision |
| Velocity Derivation | ~5% error |
| Throughput | 91k samples/sec |

---

## Examples

### Transform Derivation

```bash
python examples/demo_transform_derivation.py
# Tests: Translation, Rotation, Subset, Scaling, Reflection
# Result: 5/5 pass
```

### Causality Detection

```bash
python examples/demo_causality.py
# Separates static from dynamic points
# Result: 2/2 pass
```

### Zeiss Benchmark

```bash
python examples/benchmark_zeiss.py
# Tests: Drift, Correction, Fiber, Noise, Fiducial
# Result: 5/6 pass
```
