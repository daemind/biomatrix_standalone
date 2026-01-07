#!/usr/bin/env python3
"""
Demo: Sign Language Gesture Recognition via Geometric Signatures

This demo shows how BioMatrix can recognize gestures (like sign language letters)
using dynamic geometric signatures instead of neural networks.

Approach:
1. Hand skeleton = 21 joints in 3D (MediaPipe format)
2. Gesture = sequence of hand poses over time
3. Signature = [velocity, curvature, finger spread, topology]
4. Classification = nearest signature in library

This is "Dynamic YOLO" - recognition through geometric understanding,
not learned features.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


# ============================================================
# HAND SKELETON MODEL (MediaPipe-like)
# ============================================================

# 21 landmarks: wrist + 4 fingers × 4 joints + thumb × 4 joints
LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

# Finger tip indices
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky


def generate_base_hand():
    """Generate base hand skeleton in rest position (palm facing camera)."""
    hand = np.zeros((21, 3))
    
    # Wrist at origin
    hand[0] = [0, 0, 0]
    
    # Thumb (angled outward)
    hand[1] = [-0.3, 0.1, 0]      # CMC
    hand[2] = [-0.5, 0.3, 0]      # MCP
    hand[3] = [-0.6, 0.5, 0]      # IP
    hand[4] = [-0.65, 0.7, 0]     # TIP
    
    # Index finger
    hand[5] = [-0.15, 0.3, 0]     # MCP
    hand[6] = [-0.15, 0.5, 0]     # PIP
    hand[7] = [-0.15, 0.7, 0]     # DIP
    hand[8] = [-0.15, 0.9, 0]     # TIP
    
    # Middle finger
    hand[9] = [0, 0.35, 0]        # MCP
    hand[10] = [0, 0.55, 0]       # PIP
    hand[11] = [0, 0.75, 0]       # DIP
    hand[12] = [0, 0.95, 0]       # TIP
    
    # Ring finger
    hand[13] = [0.15, 0.3, 0]     # MCP
    hand[14] = [0.15, 0.5, 0]     # PIP
    hand[15] = [0.15, 0.7, 0]     # DIP
    hand[16] = [0.15, 0.85, 0]    # TIP
    
    # Pinky finger
    hand[17] = [0.3, 0.25, 0]     # MCP
    hand[18] = [0.3, 0.4, 0]      # PIP
    hand[19] = [0.3, 0.55, 0]     # DIP
    hand[20] = [0.3, 0.7, 0]      # TIP
    
    return hand


def generate_gesture_A(n_frames=30):
    """Generate ASL letter 'A' gesture: fist with thumb beside index."""
    frames = []
    base = generate_base_hand()
    
    for t in range(n_frames):
        hand = base.copy()
        progress = min(1.0, t / (n_frames * 0.5))
        
        # Curl all fingers into fist
        for finger_base in [5, 9, 13, 17]:  # Index, Middle, Ring, Pinky MCPs
            for i in range(4):
                idx = finger_base + i
                # Curl toward palm
                curl = progress * 0.6
                hand[idx, 1] = base[idx, 1] - curl * (i + 1) * 0.15
                hand[idx, 2] = -curl * (i + 1) * 0.1
        
        # Thumb stays beside index
        hand[1:5, 0] = base[1:5, 0] + progress * 0.1
        hand[1:5, 2] = -progress * 0.1
        
        # Add slight noise
        hand += np.random.randn(21, 3) * 0.01
        frames.append(hand)
    
    return frames


def generate_gesture_B(n_frames=30):
    """Generate ASL letter 'B' gesture: flat hand, fingers together, thumb tucked."""
    frames = []
    base = generate_base_hand()
    
    for t in range(n_frames):
        hand = base.copy()
        progress = min(1.0, t / (n_frames * 0.5))
        
        # Keep fingers straight, bring together
        for finger_base in [5, 9, 13, 17]:
            for i in range(4):
                idx = finger_base + i
                # Straighten and align
                hand[idx, 0] = hand[idx, 0] * (1 - progress * 0.5)
        
        # Tuck thumb across palm
        hand[2:5, 0] = base[2:5, 0] + progress * 0.3
        hand[2:5, 1] = base[2:5, 1] - progress * 0.2
        hand[2:5, 2] = -progress * 0.15
        
        # Add slight noise
        hand += np.random.randn(21, 3) * 0.01
        frames.append(hand)
    
    return frames


def generate_gesture_C(n_frames=30):
    """Generate ASL letter 'C' gesture: curved hand like holding a cup."""
    frames = []
    base = generate_base_hand()
    
    for t in range(n_frames):
        hand = base.copy()
        progress = min(1.0, t / (n_frames * 0.5))
        
        # Curve all fingers into C shape
        for finger_base in [1, 5, 9, 13, 17]:
            for i in range(4):
                idx = finger_base + i
                curve = progress * 0.3 * (i + 1) * 0.25
                hand[idx, 2] = -curve
                hand[idx, 0] = base[idx, 0] - curve * 0.3
        
        # Add slight noise
        hand += np.random.randn(21, 3) * 0.01
        frames.append(hand)
    
    return frames


# ============================================================
# GEOMETRIC SIGNATURE EXTRACTION
# ============================================================

def compute_signature(frames):
    """
    Compute geometric signature from gesture sequence.
    
    Signature components:
    1. Total displacement (via Procrustes)
    2. Mean finger spread (distance between tips)
    3. Curl factor (distance from tips to palm)
    4. Velocity profile (mean frame-to-frame motion)
    """
    # 1. Total displacement via Procrustes
    if len(frames) >= 2:
        op = derive_procrustes_se3(State(frames[0]), State(frames[-1]))
        displacement = np.linalg.norm(op.t) if op else 0
    else:
        displacement = 0
    
    # 2. Final finger spread (pairwise distances between tips)
    final_hand = frames[-1]
    tips = final_hand[FINGER_TIPS]
    tip_distances = distance_matrix(tips, tips)
    spread = np.mean(tip_distances[np.triu_indices(5, k=1)])
    
    # 3. Curl factor (how close tips are to wrist)
    wrist = final_hand[0]
    tip_to_wrist = np.mean([np.linalg.norm(final_hand[t] - wrist) for t in FINGER_TIPS])
    
    # 4. Velocity profile
    velocities = []
    for i in range(1, len(frames)):
        v = np.mean(np.linalg.norm(frames[i] - frames[i-1], axis=1))
        velocities.append(v)
    mean_velocity = np.mean(velocities) if velocities else 0
    
    # 5. Final pose compactness
    compactness = np.std(final_hand[:, 0]) + np.std(final_hand[:, 1])
    
    signature = np.array([
        displacement,
        spread,
        tip_to_wrist,
        mean_velocity,
        compactness
    ])
    
    return signature


def classify_gesture(signature, library):
    """
    Classify gesture by finding nearest signature in library.
    
    Uses Euclidean distance in signature space.
    """
    min_dist = float('inf')
    best_class = None
    
    for label, ref_sig in library.items():
        dist = np.linalg.norm(signature - ref_sig)
        if dist < min_dist:
            min_dist = dist
            best_class = label
    
    return best_class, min_dist


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BIOMATRIX DEMO: Sign Language Gesture Recognition")
    print("=" * 60)
    print("\nRecognizing ASL letters via geometric signatures...")
    print("No neural network. Pure geometric analysis.\n")
    
    # Generate reference gestures and build signature library
    print("Building signature library...")
    library = {}
    
    gestures = {
        'A': generate_gesture_A(30),
        'B': generate_gesture_B(30),
        'C': generate_gesture_C(30)
    }
    
    for label, frames in gestures.items():
        sig = compute_signature(frames)
        library[label] = sig
        print(f"  {label}: {sig}")
    
    # Test classification
    print("\n" + "=" * 40)
    print("CLASSIFICATION TEST")
    print("=" * 40)
    
    # Generate test gestures with noise
    test_cases = [
        ('A', generate_gesture_A(25)),
        ('B', generate_gesture_B(35)),
        ('C', generate_gesture_C(28)),
    ]
    
    correct = 0
    for true_label, frames in test_cases:
        sig = compute_signature(frames)
        pred_label, dist = classify_gesture(sig, library)
        status = "✓" if pred_label == true_label else "✗"
        if pred_label == true_label:
            correct += 1
        print(f"  {status} True: {true_label}, Predicted: {pred_label} (dist={dist:.4f})")
    
    accuracy = 100 * correct / len(test_cases)
    print(f"\nAccuracy: {accuracy:.0f}%")
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    for i, (label, frames) in enumerate(gestures.items()):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Draw final hand pose
        hand = frames[-1]
        
        # Draw joints
        ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], c='blue', s=50)
        
        # Draw finger connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for a, b in connections:
            ax.plot([hand[a, 0], hand[b, 0]], 
                    [hand[a, 1], hand[b, 1]], 
                    [hand[a, 2], hand[b, 2]], 'b-', linewidth=2)
        
        # Highlight fingertips
        tips = hand[FINGER_TIPS]
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], c='red', s=100, marker='o')
        
        ax.set_title(f'ASL Letter "{label}"')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.5, 1.5])
        ax.set_zlim([-0.5, 0.5])
    
    plt.tight_layout()
    plt.savefig('demo_sign_language.png', dpi=150)
    print("\nPlot saved: demo_sign_language.png")
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("GEOMETRIC SIGNATURE COMPONENTS")
    print("=" * 60)
    print("""
Each gesture is characterized by:
1. Displacement: Total motion from start to end
2. Spread: Mean distance between fingertips
3. Tip-to-wrist: How closed the hand is
4. Velocity: Speed of gesture execution
5. Compactness: Spatial extent of final pose

Classification: Nearest neighbor in signature space.
No training required - just geometric measurement.

This approach can be extended to:
- Full ASL alphabet (26 letters)
- Dynamic signs (words with motion)
- Two-handed signs
- Any articulated skeleton (body, face)
""")


if __name__ == "__main__":
    main()
