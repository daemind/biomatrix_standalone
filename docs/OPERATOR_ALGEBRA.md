# Operator Algebra Documentation

## Overview

The Operator Algebra layer provides symbolic representation and algebraic manipulation for BioMatrix operators. It enables:

- **Symbolic notation** for human-readable operator descriptions
- **Category inference** for composition type checking
- **Structural unification** for ARC generalization
- **Parameter extraction** for analyzing operator variations

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   SYMBOLIC LAYER                             │
├──────────────────────────────────────────────────────────────┤
│  to_symbolic()       : T(1,0), Tile(3), (A ∪ B), Φ → B → Π  │
│  structural_pattern  : T(...), Φ(...) → B → Π(...)          │
│  extract_parameters  : translation, n_tiles, dim, value     │
│  compare_parameters  : constant vs varying                   │
│  unify_solutions     : structural_unified=True/False        │
│  analyze_arc_task    : full pipeline                         │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                 N-DIMENSIONAL ALGEBRAIC CORE                 │
├──────────────────────────────────────────────────────────────┤
│  State: np.ndarray [N, D] - Dimension agnostic              │
│  Operators: apply(state) → state                            │
│  Lifting kernels: Φ: R^D → R^{D+K}                          │
│  Bijection: tensor algebra                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Operator Categories

```python
from biomatrix.core.base import OperatorCategory
```

| Category | Description | Example |
|----------|-------------|---------|
| `BIJECTION` | Mass-preserving, invertible | AffineTransform |
| `INJECTION` | Reduces mass | Filter, Delete |
| `SURJECTION` | Increases mass | Tiling, Union |
| `PROJECTION` | Idempotent (T² = T) | Select |
| `IDENTITY` | T(S) = S | IdentityOperator |

---

## Symbolic Notation

### Core Operators

```python
from biomatrix.core.transform import AffineTransform

T = AffineTransform.translate([3, 5])
T.to_symbolic()  # "T(3.00, 5.00)"

R = AffineTransform.rotation_plane(np.pi/4, 0, 1, n_dims=2)
R.to_symbolic()  # "R(45.0°)"

S = AffineTransform.scale(2.0, n_dims=2)
S.to_symbolic()  # "S(2.00)"
```

### Tiling Operators

```python
from biomatrix.core.operators.replication import TilingOperator, LinearSequenceOperator

tile = TilingOperator(translations=np.array([[0,0],[5,0],[10,0]]))
tile.to_symbolic()  # "Tile(3)"

seq = LinearSequenceOperator(count=5, step=np.array([2.0, 0.0]))
seq.to_symbolic()  # "Σ_5τ(2.0, 0.0)"
```

### Logic Operators

```python
from biomatrix.core.operators.logic import UnionOperator, IntersectionOperator

union = UnionOperator(operands=[T1, T2])
union.to_symbolic()  # "(T(1.00, 0.00) ∪ T(0.00, 1.00))"

inter = IntersectionOperator(operands=[T1, T2])
inter.to_symbolic()  # "(T(1.00, 0.00) ∩ T(0.00, 1.00))"
```

### LiftedTransform

```python
from biomatrix.core.operators.algebra import LiftedTransform

lifted.to_symbolic()  # "Φ(topology) → B(t=3.7) → Π(2 vals)"
```

---

## Algebraic Properties

### Operators expose algebraic properties:

```python
op.category         # OperatorCategory enum
op.is_identity      # bool: T(S) = S
op.is_invertible    # bool: T⁻¹ exists
op.is_idempotent    # bool: T² = T
op.preserves_mass   # bool: |T(S)| = |S|
```

### For invertible operators:

```python
T = AffineTransform.translate([1, 2])
T_inv = T.inverse()
T_inv.to_symbolic()  # "T(-1.00, -2.00)"
```

---

## Composition

### Using @ operator:

```python
T1 = AffineTransform.translate([1, 0])
T2 = AffineTransform.rotate(np.pi/2)
composed = T1 @ T2  # T1 ∘ T2

composed.to_symbolic()  # "T(1.00, 0.00) ∘ R(90.0°)"
```

### Simplification:

```python
T1 = AffineTransform.translate([1, 0])
T2 = AffineTransform.translate([0, 2])
T3 = AffineTransform.translate([3, 3])

composed = T1 @ T2 @ T3
simplified = composed.simplify()
simplified.to_symbolic()  # "T(4.00, 5.00)" - merged!
```

---

## ARC Analysis Functions

### unify_solutions()

Check if derived operators share structural form:

```python
from biomatrix.core.base import unify_solutions

T1 = AffineTransform.translate([1, 0])
T2 = AffineTransform.translate([2, 0])
T3 = AffineTransform.translate([3, 0])

result = unify_solutions([T1, T2, T3])
# {
#   'unified': False,              # Exact params differ
#   'structural_unified': True,    # Same structure!
#   'structural_pattern': 'T(...)',
#   'generalizes': True
# }
```

### compare_parameters()

Find what's constant vs varying across operators:

```python
from biomatrix.core.base import compare_parameters

result = compare_parameters([T1, T2, T3])
# {
#   'constant': {'linear_det': 1.0},
#   'varying': {'translation': [[1,0], [2,0], [3,0]]}
# }
```

### analyze_arc_task()

Full ARC task analysis pipeline:

```python
from biomatrix.core.base import analyze_arc_task

result = analyze_arc_task([
    (s_in1, s_out1),  # Training pair 1
    (s_in2, s_out2),  # Training pair 2
    (s_in3, s_out3),  # Training pair 3
])

if result['generalizes']:
    # Apply to test
    test_output = result['operators'][0].apply(test_input)
```

---

## Commutativity

Check if operators commute:

```python
from biomatrix.core.base import commutes

T1 = AffineTransform.translate([1, 0, 0])
T2 = AffineTransform.translate([0, 1, 0])
R = AffineTransform.rotation_plane(np.pi/4, 0, 1, n_dims=3)

commutes(T1, T2)  # True - translations commute
commutes(T1, R)   # False - rotation breaks commutativity
```

---

## Composition Analysis

```python
from biomatrix.core.base import analyze_composition, infer_result_category

ops = [T1, tile, select]
analysis = analyze_composition(ops)
# {
#   'n_ops': 3,
#   'categories': ['BIJECTION', 'SURJECTION', 'PROJECTION'],
#   'is_bijective': False,
#   'is_invertible': False,
#   'preserves_mass': False,
#   'symbolic': 'σ(d2=0.0) ∘ Tile(2) ∘ T(1,0,0)'
# }

result_cat = infer_result_category(ops)
# OperatorCategory.SURJECTION
```

---

## Symbol Reference

| Operator | Symbol | Example |
|----------|--------|---------|
| Translation | `T(...)` | `T(3.00, 5.00)` |
| Rotation | `R(...)` | `R(45.0°)` |
| Scale | `S(...)` | `S(2.00)` |
| Identity | `Id` | `Id` |
| Tiling | `Tile(...)` | `Tile(3)` |
| Sequence | `Σ_nτ(...)` | `Σ_5τ(2.0, 0.0)` |
| Kronecker | `K(...)` | `K(2.0, 2.0)` |
| Minkowski | `⊕(...)` | `⊕(3)` |
| Select | `σ(...)` | `σ(d2=3.0)` |
| Filter | `π(...)` | `π(d1>=5.0)` |
| Delete | `δ(...)` | `δ(del d2=0.0)` |
| Lift | `Φ(...)` | `Φ(topology)` |
| Bijection | `B(...)` | `B(t=3.7)` |
| Slice | `Π(...)` | `Π(→3D)` |
| Union | `(A ∪ B)` | `(T(1,0) ∪ Id)` |
| Intersection | `(A ∩ B)` | `(T(1,0) ∩ T(0,1))` |
| Difference | `(A \ B)` | `(T(1,0) \ Id)` |
