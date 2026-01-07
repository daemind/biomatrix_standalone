# -*- coding: utf-8 -*-
"""
Zeiss ZEN Protocol Handler

Implements communication with Zeiss ZEN software via:
- TCP/IP socket connection
- ZEN API commands
- Frame data parsing
"""

import socket
import struct
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import threading
import queue
import time


@dataclass
class ZENFrame:
    """Frame data from ZEN."""
    timestamp: float
    width: int
    height: int
    channels: int
    data: np.ndarray
    metadata: Dict[str, Any]


class ZENProtocol:
    """
    Zeiss ZEN communication protocol.
    
    Protocol format (simplified):
    - Header: 16 bytes (magic, version, type, size)
    - Metadata: JSON-encoded
    - Data: Raw pixel data
    """
    
    MAGIC = b'ZEN\x00'
    VERSION = 1
    
    # Message types
    MSG_CONNECT = 0x01
    MSG_DISCONNECT = 0x02
    MSG_START_STREAM = 0x10
    MSG_STOP_STREAM = 0x11
    MSG_FRAME = 0x20
    MSG_METADATA = 0x30
    MSG_ACK = 0xF0
    MSG_ERROR = 0xFF
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self._recv_thread = None
        self._frame_queue = queue.Queue(maxsize=100)
        
    def connect(self) -> bool:
        """Connect to ZEN server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            
            # Send connect message
            self._send_message(self.MSG_CONNECT, b'')
            
            # Wait for ACK
            msg_type, _ = self._recv_message()
            if msg_type == self.MSG_ACK:
                self.connected = True
                return True
                
        except Exception as e:
            print(f"ZEN connection failed: {e}")
            
        return False
        
    def disconnect(self):
        """Disconnect from ZEN server."""
        if self.connected:
            try:
                self._send_message(self.MSG_DISCONNECT, b'')
            except:
                pass
        if self.socket:
            self.socket.close()
        self.connected = False
        
    def start_stream(self):
        """Start frame streaming."""
        if not self.connected:
            return False
            
        self._send_message(self.MSG_START_STREAM, b'')
        
        # Start receiver thread
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
        
        return True
        
    def stop_stream(self):
        """Stop frame streaming."""
        if self.connected:
            self._send_message(self.MSG_STOP_STREAM, b'')
            
    def get_frame(self, timeout: float = 1.0) -> Optional[ZENFrame]:
        """Get next frame from queue."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _send_message(self, msg_type: int, data: bytes):
        """Send message to ZEN."""
        header = struct.pack('>4sBBH', self.MAGIC, self.VERSION, msg_type, len(data))
        self.socket.sendall(header + data)
        
    def _recv_message(self) -> Tuple[int, bytes]:
        """Receive message from ZEN."""
        header = self._recv_exact(8)
        magic, version, msg_type, size = struct.unpack('>4sBBH', header)
        
        if magic != self.MAGIC:
            raise ValueError("Invalid ZEN message")
            
        data = self._recv_exact(size) if size > 0 else b''
        return msg_type, data
        
    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
        
    def _receive_loop(self):
        """Background frame receiver."""
        while self.connected:
            try:
                msg_type, data = self._recv_message()
                
                if msg_type == self.MSG_FRAME:
                    frame = self._parse_frame(data)
                    if frame:
                        try:
                            self._frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Drop oldest frames
                            
            except Exception as e:
                if self.connected:
                    print(f"ZEN receive error: {e}")
                break
                
    def _parse_frame(self, data: bytes) -> Optional[ZENFrame]:
        """Parse frame data."""
        try:
            # Header: timestamp (8), width (4), height (4), channels (4)
            header_size = 20
            timestamp, width, height, channels = struct.unpack('>dIII', data[:header_size])
            
            # Pixel data (float32)
            pixel_data = np.frombuffer(data[header_size:], dtype=np.float32)
            pixel_data = pixel_data.reshape((height, width, channels))
            
            return ZENFrame(
                timestamp=timestamp,
                width=width,
                height=height,
                channels=channels,
                data=pixel_data,
                metadata={}
            )
            
        except Exception as e:
            print(f"Frame parse error: {e}")
            return None


class ZENMockServer:
    """
    Mock ZEN server for testing.
    
    Simulates ZEN frame streaming with synthetic SMLM data.
    """
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.server = None
        self._running = False
        
    def start(self):
        """Start mock server."""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('localhost', self.port))
        self.server.listen(1)
        
        self._running = True
        threading.Thread(target=self._serve, daemon=True).start()
        
    def stop(self):
        """Stop mock server."""
        self._running = False
        if self.server:
            self.server.close()
            
    def _serve(self):
        """Main server loop."""
        while self._running:
            try:
                client, addr = self.server.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True
                ).start()
            except:
                break
                
    def _handle_client(self, client: socket.socket):
        """Handle client connection."""
        streaming = False
        
        while self._running:
            try:
                header = client.recv(8)
                if not header:
                    break
                    
                magic, version, msg_type, size = struct.unpack('>4sBBH', header)
                data = client.recv(size) if size > 0 else b''
                
                if msg_type == ZENProtocol.MSG_CONNECT:
                    # Send ACK
                    ack = struct.pack('>4sBBH', ZENProtocol.MAGIC, 1, ZENProtocol.MSG_ACK, 0)
                    client.sendall(ack)
                    
                elif msg_type == ZENProtocol.MSG_START_STREAM:
                    streaming = True
                    # Start sending frames
                    while streaming and self._running:
                        frame_data = self._generate_frame()
                        header = struct.pack(
                            '>4sBBH',
                            ZENProtocol.MAGIC, 1,
                            ZENProtocol.MSG_FRAME,
                            len(frame_data)
                        )
                        client.sendall(header + frame_data)
                        time.sleep(0.033)  # ~30 FPS
                        
                elif msg_type == ZENProtocol.MSG_STOP_STREAM:
                    streaming = False
                    
                elif msg_type == ZENProtocol.MSG_DISCONNECT:
                    break
                    
            except:
                break
                
        client.close()
        
    def _generate_frame(self) -> bytes:
        """Generate synthetic SMLM frame."""
        width, height, channels = 64, 64, 1
        
        # Synthetic molecules
        n_molecules = np.random.randint(10, 50)
        frame = np.zeros((height, width, channels), dtype=np.float32)
        
        for _ in range(n_molecules):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Gaussian PSF
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        frame[ny, nx, 0] += intensity * np.exp(-(dx**2 + dy**2) / 2)
        
        # Pack frame
        timestamp = time.time()
        header = struct.pack('>dIII', timestamp, width, height, channels)
        return header + frame.tobytes()
