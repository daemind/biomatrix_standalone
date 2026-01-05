# -*- coding: utf-8 -*-
"""
Zeiss Live Tracking Plugin

Real-time SMLM and DNA fiber tracking for Zeiss microscopes.
"""

from .stream import FrameStream
from .tracker import LiveTracker
from .drift import DriftCorrector
from .fiber import FiberLinker

__all__ = [
    'FrameStream',
    'LiveTracker', 
    'DriftCorrector',
    'FiberLinker',
]


class ZeissPlugin:
    """
    Main plugin interface for Zeiss microscopy integration.
    
    Usage:
        plugin = ZeissPlugin(source="zeiss://localhost:5000")
        plugin.start(mode="smlm", drift_correction=True)
    """
    
    def __init__(self, source: str = None):
        self.source = source
        self.stream = None
        self.tracker = LiveTracker()
        self.drift = DriftCorrector()
        self.fiber = FiberLinker()
        self._callbacks = {}
        
    def start(self, mode: str = "smlm", drift_correction: bool = True):
        """Start live tracking."""
        self.stream = FrameStream(self.source)
        
        def process_frame(pts):
            # Track
            op = self.tracker.track(pts)
            
            # Drift correction
            corrected = self.drift.correct(pts) if drift_correction else pts
            
            # Mode-specific processing
            if mode == "fiber":
                self.fiber.process(corrected)
            
            # Callbacks
            if "frame" in self._callbacks:
                self._callbacks["frame"](corrected, op)
                
        self.stream.on_frame(process_frame)
        
    def stop(self):
        """Stop tracking."""
        if self.stream:
            self.stream.stop()
            
    def on_drift(self, callback):
        """Register drift callback."""
        self._callbacks["drift"] = callback
        
    def on_fiber(self, callback):
        """Register fiber callback."""
        self._callbacks["fiber"] = callback
        
    def on_frame(self, callback):
        """Register frame callback."""
        self._callbacks["frame"] = callback
