#!/usr/bin/env python3
"""
Benchmark sémantique pour ARC - Test de généralisation
"""

import json
import numpy as np
import os
import sys

# Add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomatrix.core.state import State
from biomatrix.core.graph.semantic import learn_global_mapping


def grid_to_state(grid):
    """Convert ARC grid to State."""
    arr = np.array(grid)
    coords = np.argwhere(arr >= 0).astype(float)
    colors = arr[arr >= 0].reshape(-1, 1).astype(float)
    return State(np.hstack([coords, colors]))


def test_bijection_solver(task_path):
    """Test bijection-based solver on a task."""
    with open(task_path) as f:
        task = json.load(f)
    
    training = [(grid_to_state(p['input']), grid_to_state(p['output'])) 
                for p in task['train']]
    
    if 'test' not in task or not task['test']:
        return None
    if 'output' not in task['test'][0]:
        return None
    
    # Learn bijection
    learned = learn_global_mapping(training)
    color_map = learned['color_map']
    
    if not color_map:
        return {'status': 'no_mapping', 'accuracy': 0}
    
    test_in = np.array(task['test'][0]['input'])
    test_exp = np.array(task['test'][0]['output'])
    
    # Apply bijection
    correct = 0
    total = test_in.size
    
    for i in range(test_in.shape[0]):
        for j in range(test_in.shape[1]):
            c_in = float(test_in[i, j])
            c_exp = test_exp[i, j]
            c_pred = color_map.get(c_in, c_in)
            if c_pred == c_exp:
                correct += 1
    
    return {
        'status': 'tested',
        'accuracy': correct / total,
        'n_mappings': len(color_map),
        'consistent': learned['is_consistent'],
        'conflicts': len(learned['conflicts'])
    }


def main():
    data_dir = '/Users/morad/Projets/bioMatrix-MVA/biomatrix/data/training/grid_tasks2'
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    tasks = sorted(os.listdir(data_dir))[:30]
    
    results = {'perfect': [], 'partial': [], 'failed': []}
    
    print("=" * 60)
    print("SEMANTIC BIJECTION BENCHMARK")
    print("=" * 60)
    print()
    
    for task_file in tasks:
        try:
            result = test_bijection_solver(os.path.join(data_dir, task_file))
            if result is None:
                continue
            
            acc = result['accuracy']
            if acc == 1.0:
                results['perfect'].append(task_file)
                print(f"✓ {task_file}: 100% ({result['n_mappings']} mappings)")
            elif acc > 0.5:
                results['partial'].append((task_file, acc))
                print(f"~ {task_file}: {acc*100:.0f}%")
            else:
                results['failed'].append((task_file, acc))
        except Exception as e:
            print(f"! {task_file}: error - {str(e)[:40]}")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Perfect (100%): {len(results['perfect'])}")
    print(f"Partial (>50%): {len(results['partial'])}")
    print(f"Failed:         {len(results['failed'])}")


if __name__ == '__main__':
    main()
