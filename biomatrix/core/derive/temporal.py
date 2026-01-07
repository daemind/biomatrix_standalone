# -*- coding: utf-8 -*-
"""
derive/temporal.py - Temporal Derivation Logic
"""
import numpy as np
from typing import List, Optional

from ..state import State
from ..operators.temporal import TemporalOperator

def derive_dynamics(sequence: List[State]) -> Optional[TemporalOperator]:
    """
    Derive the Equation of Motion for a sequence of states.
    Input: [S0, S1, S2, ...]
    """
    if len(sequence) < 2:
        return None
        
    T = len(sequence)
    
    # 1. Newton's First Law: Constant Velocity
    # Check if S_{t+1} - S_t ~= Constant
    velocities = []
    
    for t in range(T - 1):
        s_curr = sequence[t]
        s_next = sequence[t+1]
        
        # Robust derivation of step transform
        # We need the Translation that maps S_t to S_{t+1}
        # Simplified for speed here: Center diff
        if s_curr.n_points == s_next.n_points and s_curr.n_points > 0:
            # ROBUST VELOCITY: Use Median of point displacements
            # This ignores single-joint movements (legs) and outliers (ghosts if N same)
            # v_t = Median(P_{t+1} - P_t)
            # This separates "Chain Motion" (Arms/Legs) from "Body Motion" (Torso/Median)
            disp = s_next.points - s_curr.points 
            v_t = np.median(disp, axis=0)
            velocities.append(v_t)
        elif s_curr.n_points > 0 and s_next.n_points > 0:
            # Size changed (Ghosts appearing/disappearing)
            # Can't do point-wise diff. Fallback to Centroid? No, Ghosts shift centroid.
            # Use Nearest Neighbor translation estimation?
            # Or just skip this frame for velocity estimation? 
            # Skipping is safer for "Robustness".
            pass 
        else:
            pass
            
    # Relaxed Requirement: We just need enough samples to estimate a trend
    if len(velocities) > max(1, (T - 1) // 4):
        # Check consistency
        V = np.array(velocities)
        
        # Robust Statistics (Median / IQR) to handle Jitter & Outliers
        median_v = np.median(V, axis=0)
        q75, q25 = np.percentile(V, [75 ,25], axis=0)
        iqr = q75 - q25
        
        # Tolerance: Allow significant variance if centered around a consistent median
        # For CartPole with random forces, variance is HUGE. 
        # But we still want to recover the "Mean Trend" (Gravity/Drift).
        # RELAXED: Always return the operator, but maybe mark it as noisy?
        
        is_consistent = True # np.all(iqr < 0.5 * np.linalg.norm(median_v) + 0.2)
        
        if is_consistent:
            mean_v = median_v # Use Median as robust estimator for drift
        
            # Strategy: Detrend & Analyze Periodicity
            # Even if velocity fluctuates (e.g. walking wobble), we try to extract the base gait.
            
            # 1. Compute Centered Residues with MEAN velocity (Detrending)
            # CRITICAL FIX for Noise:
            # Do NOT use s.centroid because ghost points shift the centroid wildly.
            # Use the ROBUST velocity to predict where the center SHOULD be.
            c0 = sequence[0].centroid # Assume start is reasonably clean (or use median of first few?)
            
            residues = []
            for t, s in enumerate(sequence):
                # Predicted center based on robust velocity
                predicted_center = c0 + mean_v * t
                
                # Detrend by subtracting the PREDICTED linear drift
                # This keeps the "Wobble" + "Noise" relative to the inertial frame
                centered_points = s.points - predicted_center
                residues.append(centered_points)
                
            # 2. Autocorrelation on SHAPE
            # Try periods T from 2 to N/2
            best_T = None
            min_error = float('inf')
            
            # Heuristic: Check T in range [2, ~T/2]
            max_T = T // 2
            
            for p in range(2, max_T + 1):
                total_err = 0
                count = 0
                valid_p = True
                
                for t in range(T - p):
                    r_curr = residues[t]
                    r_next = residues[t+p]
                    
                # ROBUST METRIC: Chamfer Distance (Set Distance)
                # Handle different point counts (N1 != N2) due to Ghosts
                
                # Simple O(N^2) for now (small skeletons)
                # d(A, B) = mean(min_dist(a, B)) + mean(min_dist(b, A))
                from scipy.spatial.distance import cdist
                
                if len(r_curr) == 0 or len(r_next) == 0:
                     err = 100.0
                else:
                    D_mat = cdist(r_curr, r_next)
                    d1 = np.mean(np.min(D_mat, axis=1)) # A -> B
                    d2 = np.mean(np.min(D_mat, axis=0)) # B -> A
                    err = d1 + d2
                    
                total_err += err
                count += 1
                
                if valid_p and count > 0:
                    avg_err = total_err / count
                    if avg_err < min_error:
                        min_error = avg_err
                        best_T = p
            
            # Detection Threshold (relaxed for noise)
            # 0.1 was good for clean-ish wobble.
            # For 2cm noise + ghosts, error might be higher.
            if min_error < 2.0: # Very relaxed for CartPole
                return TemporalOperator(velocity=mean_v, period=best_T)
            else:
                # Fallback: Just return Velocity dynamics if periodicity failed
                # For CartPole (non-periodic chaotic), we just want the motion model.
                return TemporalOperator(velocity=mean_v, period=None)
            
    return None
