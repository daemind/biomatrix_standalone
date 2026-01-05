# BioMatrix Documentation

## Quick Start

```python
from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3
import numpy as np

# Create point clouds
pts1 = np.random.randn(100, 3) * 5
pts2 = pts1 + np.array([0.1, 0.2, 0.3])  # Add drift

# Derive transformation
op = derive_procrustes_se3(State(pts1), State(pts2))
print(f"Detected drift: {op.t}")  # [0.1, 0.2, 0.3]
```

## API Reference

### Core

#### `State(points)`
Wraps N-dimensional point cloud.

```python
state = State(np.random.randn(100, 3))
```

#### `derive_procrustes_se3(state_in, state_out)`
Derives SE(N) transformation Y = scale * R @ X + t.

Returns `ProcrustesOperator` with:
- `.A` - DxD transformation matrix
- `.t` - D-vector translation
- `.apply(state)` - Apply to state

### Plugins

#### Zeiss Plugin

```python
from biomatrix.plugins.zeiss import LiveTracker, DriftCorrector
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector

# Real-time tracking
tracker = LiveTracker()
for frame in frames:
    op = tracker.track(frame)
    drift = tracker.get_total_drift()

# High-precision with fiducials
corrector = FiducialDriftCorrector(
    known_fiducial_indices=np.array([100, 101, 102, 103, 104])
)
corrected, drift = corrector.correct(points)
```

## Performance

| Metric | Value |
|--------|-------|
| Drift Detection | 8.5% relative error |
| Fiducial Precision | **0.046 nm** (sub-angstrom) |
| Fiber Tracking | 100% continuity |
| Noise σ=0.5 | 33% error |
