#!/usr/bin/env python3
"""Test causal group detection on ARC task."""

import json
import numpy as np
import sys
sys.path.insert(0, '/Users/morad/Projets/bioMatrix-MVA/biomatrix/biomatrix_standalone')

from biomatrix.core.state import State
from biomatrix.core.graph.semantic import detect_causal_groups, solve_arc_with_causal_groups

def grid_to_state(g):
    arr = np.array(g)
    coords = np.argwhere(arr >= 0).astype(float)
    return State(np.hstack([coords, arr[arr >= 0].reshape(-1, 1).astype(float)]))

# Test on 009d5c81
path = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2/009d5c81.json'
with open(path) as f:
    task = json.load(f)

training = [(grid_to_state(p['input']), grid_to_state(p['output'])) for p in task['train']]
test_in = grid_to_state(task['test'][0]['input'])
test_exp = np.array(task['test'][0]['output'])

# Detect groups
detected = detect_causal_groups(training)
print(f'Detected {detected["n_groups"]} causal groups:')
for sig, profile in detected['profiles'].items():
    print(f'  {int(sig[0])}→{int(sig[1])}: count={profile["count"]}, n_pts={profile["n_points_range"]}')

# Solve
predicted, explanation = solve_arc_with_causal_groups(training, test_in)
print(explanation)

# Check accuracy
pred_grid = np.zeros_like(test_exp, dtype=float)
for pt in predicted.points:
    i, j = int(pt[0]), int(pt[1])
    if 0 <= i < pred_grid.shape[0] and 0 <= j < pred_grid.shape[1]:
        pred_grid[i, j] = pt[2]

acc = np.sum(pred_grid == test_exp) / test_exp.size
print(f'\nAccuracy: {acc*100:.1f}%')
