#!/usr/bin/env python3
"""Benchmark for unified affine solver."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from biomatrix.core.state import State
from biomatrix.core.graph.semantic import solve_with_affine_composition
from biomatrix.core.graph.semantic import solve_with_affine_composition


def grid_to_state(g):
    arr = np.array(g)
    coords = np.argwhere(arr >= 0).astype(float)
    return State(np.hstack([coords, arr[arr >= 0].reshape(-1, 1).astype(float)]))


def main():
    data_dir = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2'
    tasks = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:30]
    
    results = []
    for task_file in tasks:
        try:
            with open(os.path.join(data_dir, task_file)) as f:
                task = json.load(f)
            
            training = [(grid_to_state(p['input']), grid_to_state(p['output'])) 
                       for p in task['train']]
            test_in = grid_to_state(task['test'][0]['input'])
            test_exp = np.array(task['test'][0]['output'])
            
            predicted, _ = solve_with_affine_composition(training, test_in)
            
            pred_grid = np.zeros_like(test_exp, dtype=float)
            for pt in predicted.points:
                i, j = int(np.round(pt[0])), int(np.round(pt[1]))
                if 0 <= i < pred_grid.shape[0] and 0 <= j < pred_grid.shape[1]:
                    pred_grid[i, j] = int(np.round(pt[2]))
            
            acc = np.sum(pred_grid == test_exp) / test_exp.size
            sym = '✓' if acc == 1.0 else ('◐' if acc >= 0.5 else '✗')
            print(f'{sym} {task_file}: {acc*100:.0f}%')
            results.append(acc)
        except Exception as e:
            print(f'! {task_file}: error - {str(e)[:40]}')
            results.append(0)
    
    print(f'\n=== SUMMARY ===')
    print(f'Perfect: {sum(1 for a in results if a == 1.0)}/30')
    print(f'Average: {np.mean(results)*100:.1f}%')


if __name__ == '__main__':
    main()
