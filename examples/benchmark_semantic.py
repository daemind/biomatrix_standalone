#!/usr/bin/env python3
"""
Benchmark sémantique pour ARC - Test de généralisation avec context-aware mapping
"""

import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomatrix.core.state import State
from biomatrix.core.topology import partition_by_connectivity


def grid_to_state(grid):
    arr = np.array(grid)
    coords = np.argwhere(arr >= 0).astype(float)
    colors = arr[arr >= 0].reshape(-1, 1).astype(float)
    return State(np.hstack([coords, colors]))


def size_bin(n):
    if n <= 5: return 'tiny'
    elif n <= 15: return 'small'
    elif n <= 30: return 'medium'
    else: return 'large'


def test_context_aware(task_path):
    """Test context-aware solver on a task."""
    with open(task_path) as f:
        task = json.load(f)
    
    if 'test' not in task or not task['test'] or 'output' not in task['test'][0]:
        return None
    
    # Learn context mapping from training
    ctx_map = {}
    for pair in task['train']:
        s_in = grid_to_state(pair['input'])
        s_out = grid_to_state(pair['output'])
        objs_in = partition_by_connectivity(s_in)
        objs_out = partition_by_connectivity(s_out)
        for oi, oo in zip(objs_in, objs_out):
            if oi.n_dims >= 3 and oo.n_dims >= 3:
                c_in = int(np.round(oi.points[0, 2], 0))
                c_out = int(np.round(oo.points[0, 2], 0))
                if c_in != c_out:
                    ctx_map[(c_in, size_bin(oi.n_points))] = c_out
    
    if not ctx_map:
        return {'status': 'no_mapping', 'accuracy': 0}
    
    # Apply to test
    test_grid = np.array(task['test'][0]['input']).astype(float)
    test_exp = np.array(task['test'][0]['output'])
    
    test_state = grid_to_state(task['test'][0]['input'])
    for obj in partition_by_connectivity(test_state):
        key = (int(np.round(obj.points[0, 2], 0)), size_bin(obj.n_points))
        if key in ctx_map:
            for pt in obj.points:
                test_grid[int(pt[0]), int(pt[1])] = ctx_map[key]
    
    correct = np.sum(test_grid == test_exp)
    return {'status': 'tested', 'accuracy': correct / test_exp.size, 'n_mappings': len(ctx_map)}


def main():
    data_dir = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2'
    tasks = sorted(os.listdir(data_dir))[:30]
    
    results = {'perfect': [], 'partial': [], 'failed': []}
    
    print("=" * 60)
    print("CONTEXT-AWARE SEMANTIC BENCHMARK")
    print("=" * 60)
    print()
    
    for task_file in tasks:
        try:
            result = test_context_aware(os.path.join(data_dir, task_file))
            if result is None:
                continue
            
            acc = result['accuracy']
            if acc == 1.0:
                results['perfect'].append(task_file)
                print(f"✓ {task_file}: 100% ({result['n_mappings']} ctx mappings)")
            elif acc > 0.5:
                results['partial'].append((task_file, acc))
                print(f"~ {task_file}: {acc*100:.0f}%")
            else:
                results['failed'].append((task_file, acc))
        except Exception as e:
            print(f"! {task_file}: error")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Perfect (100%): {len(results['perfect'])}")
    print(f"Partial (>50%): {len(results['partial'])}")
    print(f"Failed:         {len(results['failed'])}")


if __name__ == '__main__':
    main()
