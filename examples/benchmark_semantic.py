#!/usr/bin/env python3
"""
Benchmark sémantique pour ARC - Test avec causal groups TYPE-based
"""

import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomatrix.core.state import State
from biomatrix.core.graph.semantic import solve_arc_with_causal_groups


def grid_to_state(grid):
    arr = np.array(grid)
    coords = np.argwhere(arr >= 0).astype(float)
    colors = arr[arr >= 0].reshape(-1, 1).astype(float)
    return State(np.hstack([coords, colors]))


def test_causal_groups(task_path):
    """Test causal groups solver on a task."""
    with open(task_path) as f:
        task = json.load(f)
    
    if 'test' not in task or not task['test'] or 'output' not in task['test'][0]:
        return None
    
    training = [(grid_to_state(p['input']), grid_to_state(p['output'])) 
                for p in task['train']]
    test_in = grid_to_state(task['test'][0]['input'])
    test_exp = np.array(task['test'][0]['output'])
    
    # Solve with causal groups
    predicted, explanation = solve_arc_with_causal_groups(training, test_in)
    
    # Build prediction grid
    pred_grid = np.zeros_like(test_exp, dtype=float)
    for pt in predicted.points:
        i, j = int(pt[0]), int(pt[1])
        if 0 <= i < pred_grid.shape[0] and 0 <= j < pred_grid.shape[1]:
            pred_grid[i, j] = pt[2]
    
    correct = np.sum(pred_grid == test_exp)
    return {
        'status': 'tested', 
        'accuracy': correct / test_exp.size
    }


def main():
    data_dir = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2'
    tasks = sorted(os.listdir(data_dir))[:30]
    
    results = {'perfect': [], 'partial': [], 'low': [], 'failed': []}
    
    print("=" * 60)
    print("CAUSAL GROUPS BENCHMARK (TYPE-BASED)")
    print("=" * 60)
    print()
    
    for task_file in tasks:
        try:
            result = test_causal_groups(os.path.join(data_dir, task_file))
            if result is None:
                continue
            
            acc = result['accuracy']
            if acc == 1.0:
                results['perfect'].append(task_file)
                print(f"✓ {task_file}: 100%")
            elif acc >= 0.9:
                results['partial'].append((task_file, acc))
                print(f"◐ {task_file}: {acc*100:.0f}%")
            elif acc > 0.5:
                results['low'].append((task_file, acc))
                print(f"~ {task_file}: {acc*100:.0f}%")
            else:
                results['failed'].append((task_file, acc))
                print(f"✗ {task_file}: {acc*100:.0f}%")
        except Exception as e:
            print(f"! {task_file}: error - {str(e)[:40]}")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Perfect (100%):  {len(results['perfect'])}")
    print(f"High (>=90%):    {len(results['partial'])}")
    print(f"Partial (>50%):  {len(results['low'])}")
    print(f"Failed (<=50%):  {len(results['failed'])}")
    
    total = len(results['perfect']) + len(results['partial']) + len(results['low']) + len(results['failed'])
    avg = np.mean([1.0] * len(results['perfect']) + 
                  [a for _, a in results['partial']] +
                  [a for _, a in results['low']] +
                  [a for _, a in results['failed']]) if total > 0 else 0
    print(f"\nAverage accuracy: {avg*100:.1f}%")


if __name__ == '__main__':
    main()
