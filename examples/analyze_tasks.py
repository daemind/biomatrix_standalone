#!/usr/bin/env python3
"""Analyze failing tasks to understand required decomposition."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from biomatrix.core.state import State


def grid_to_state(g):
    arr = np.array(g)
    coords = np.argwhere(arr >= 0).astype(float)
    return State(np.hstack([coords, arr[arr >= 0].reshape(-1, 1).astype(float)]))


def main():
    data_dir = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2'
    tasks = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:30]
    
    # Group tasks by transformation type
    expansion = []  # n_out > n_in
    contraction = []  # n_out < n_in
    same_shape = []  # n_out == n_in
    
    for task_file in tasks:
        with open(os.path.join(data_dir, task_file)) as f:
            task = json.load(f)
        
        s_in = grid_to_state(task['train'][0]['input'])
        s_out = grid_to_state(task['train'][0]['output'])
        
        ratio = s_out.n_points / s_in.n_points if s_in.n_points > 0 else 0
        
        if ratio > 1.1:
            expansion.append((task_file, ratio))
        elif ratio < 0.9:
            contraction.append((task_file, ratio))
        else:
            same_shape.append((task_file, ratio))
    
    print(f"=== Task Type Analysis ===\n")
    print(f"EXPANSION (n_out > n_in): {len(expansion)}")
    for f, r in expansion[:5]:
        print(f"  {f}: ratio={r:.2f}")
    
    print(f"\nCONTRACTION (n_out < n_in): {len(contraction)}")
    for f, r in contraction[:5]:
        print(f"  {f}: ratio={r:.2f}")
    
    print(f"\nSAME SHAPE (n_out ≈ n_in): {len(same_shape)}")
    for f, r in same_shape[:5]:
        print(f"  {f}: ratio={r:.2f}")


if __name__ == '__main__':
    main()
