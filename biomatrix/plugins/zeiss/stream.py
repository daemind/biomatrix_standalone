# -*- coding: utf-8 -*-
"""
Frame Stream - Real-time frame acquisition

Supports:
- Zeiss ZEN protocol (zeiss://)
- File sequences (file://)
- Mock data for testing
"""

import numpy as np
from typing import Callable, Optional
from collections import deque
import threading
import time


class FrameStream:
    """
    Real-time frame acquisition from various sources.
    
    N-dimensional agnostic: works with 2D or 3D point clouds.
    """
    
    def __init__(self, source: Optional[str] = None):
        """
        Args:
            source: Connection string
                - "zeiss://host:port" for Zeiss ZEN
                - "file://path/*.tif" for file sequence
                - None for mock data
        """
        self.source = source
        self._running = False
        self._callback = None
        self._thread = None
        self._frame_queue = deque(maxlen=100)
        
    def on_frame(self, callback: Callable[[np.ndarray], None]):
        """Register frame callback."""
        self._callback = callback
        self._start_acquisition()
        
    def next_frame(self) -> Optional[np.ndarray]:
        """Get next frame synchronously."""
        if self._frame_queue:
            return self._frame_queue.popleft()
        return None
        
    def _start_acquisition(self):
        """Start acquisition thread."""
        self._running = True
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()
        
    def _acquisition_loop(self):
        """Main acquisition loop."""
        while self._running:
            pts = self._acquire_frame()
            if pts is not None:
                self._frame_queue.append(pts)
                if self._callback:
                    self._callback(pts)
            time.sleep(0.033)  # ~30 FPS
            
    def _acquire_frame(self) -> Optional[np.ndarray]:
        """Acquire single frame from source."""
        if self.source is None:
            # Mock data: random point cloud with drift
            return self._generate_mock_frame()
        elif self.source.startswith("zeiss://"):
            return self._acquire_zeiss()
        elif self.source.startswith("file://"):
            return self._acquire_file()
        return None
        
    def _generate_mock_frame(self) -> np.ndarray:
        """Generate mock SMLM-like data."""
        n_pts = np.random.randint(50, 200)
        
        # Base structure (simulated molecules)
        t = np.linspace(0, 2 * np.pi, n_pts)
        pts = np.column_stack([
            np.cos(t) + np.random.randn(n_pts) * 0.1,
            np.sin(t) + np.random.randn(n_pts) * 0.1,
            t / (2 * np.pi) + np.random.randn(n_pts) * 0.05
        ])
        
        # Add thermal drift
        drift = np.array([0.001, 0.002, 0.0005]) * time.time() % 1
        pts += drift
        
        return pts
        
    def _acquire_zeiss(self) -> Optional[np.ndarray]:
        """Acquire from Zeiss ZEN (stub)."""
        # TODO: Implement Zeiss protocol
        return self._generate_mock_frame()
        
    def _acquire_file(self) -> Optional[np.ndarray]:
        """Acquire from file sequence (stub)."""
        # TODO: Implement file reading
        return self._generate_mock_frame()
        
    def stop(self):
        """Stop acquisition."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
