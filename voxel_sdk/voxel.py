# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""Voxel FileSystem interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import json
import socket
import struct
import ipaddress

try:  # Optional dependencies for visualization
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
    np = None

try:  # Optional dependency for hand pose overlays
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None


class HandPoseOverlay:
    def __init__(self):
        if mp is None:
            raise RuntimeError(
                "Hand tracking requires mediapipe. Install it with `pip install mediapipe`."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def annotate(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._drawing.draw_landmarks(
                    image, hand_landmarks, self._connections
                )
        return image

    def process(self, image):
        """Process image and return hand landmarks data."""
        if np is None or cv2 is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None
        
        h, w = image.shape[:2]
        landmarks_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z  # Relative depth (normalized)
                points.append((x, y, z))
            landmarks_data.append(points)
        return landmarks_data

    def close(self):
        self._hands.close()


def create_hand_overlay(enable: bool):
    if not enable:
        return None
    return HandPoseOverlay()


class VoxelTransport(ABC):
    """Abstract base class for device communication protocols."""
    
    @abstractmethod
    def connect(self, address: str) -> None:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        pass
    
    @abstractmethod
    def send_command(self, command: str, data: Optional[str] = None) -> Dict[str, Any]:
        pass


class VoxelFileSystem:
    """Unified interface for Voxel device filesystem operations."""
    
    def __init__(self, transport: VoxelTransport, address: Optional[str] = None):
        self.transport = transport
        self.address = address
    
    def _ensure_connected(self) -> None:
        if not self.is_connected():
            raise ConnectionError("Not connected to Voxel device")
    
    def _send_command(self, command: str, data: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_connected()
        return self.transport.send_command(command, data)
        
    def connect(self) -> None:
        if self.address:
            self.transport.connect(self.address)
        else:
            self.transport.connect("")
    
    def disconnect(self) -> None:
        self.transport.disconnect()
    
    def is_connected(self) -> bool:
        return self.transport.is_connected()
    
    def list_directory(self, path: str = "/", levels: int = 0) -> List[Dict[str, Any]]:
        response = self._send_command("list_dir", path)
        return response.get("files", [])
    
    def create_directory(self, path: str) -> Dict[str, Any]:
        return self._send_command("create_dir", path)
    
    def remove_directory(self, path: str) -> Dict[str, Any]:
        return self._send_command("remove_dir", path)
    
    def read_file(self, path: str) -> str:
        response = self._send_command("read_file", path)
        return response.get("content", "")
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        return self._send_command("write_file", f"{path}|{content}")
    
    def append_file(self, path: str, content: str) -> Dict[str, Any]:
        return self._send_command("append_file", f"{path}|{content}")
    
    def rename_file(self, old_path: str, new_path: str) -> Dict[str, Any]:
        return self._send_command("rename_file", f"{old_path},{new_path}")
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        return self._send_command("delete_file", path)
    
    def get_card_info(self) -> Dict[str, Any]:
        response = self._send_command("card_info")
        return response
    
    def file_exists(self, path: str) -> bool:
        response = self._send_command("file_exists", path)
        return response.get("exists", False)
    
    def get_file_size(self, path: str) -> Optional[int]:
        response = self._send_command("file_size", path)
        return response.get("size")
    
    def download_file(self, path: str, progress_callback=None) -> bytes:
        """Download a file from the device and return its contents as bytes."""
        self._ensure_connected()
        
        # Check if transport supports download_file method
        if hasattr(self.transport, 'download_file'):
            return self.transport.download_file(path, progress_callback)
        else:
            raise NotImplementedError("Transport does not support file download")

    def connect_wifi(self, ssid: str, password: str) -> Dict[str, Any]:
        """Connect the device to WiFi using the provided SSID and password."""
        if not ssid:
            raise ValueError("SSID cannot be empty")

        password = password or ""
        data = f"{ssid}|{password}"
        return self._send_command("connectWifi", data)

    def disconnect_wifi(self) -> Dict[str, Any]:
        """Disconnect the device from WiFi."""
        return self._send_command("disconnectWifi")

    def get_wifi_status(self) -> Dict[str, Any]:
        """Retrieve current WiFi connection status."""
        return self._send_command("wifiStatus")
    
    def ping_host(self, host: str, count: int = 4) -> Dict[str, Any]:
        """Ping a host from the device."""
        if not host:
            raise ValueError("Host cannot be empty")

        count = max(1, min(count, 10))
        data = f"{host}|{count}" if count else host
        return self._send_command("ping_host", data)

    def start_rdmp_stream(self, host: str, port: int = 9000, quality: Optional[int] = None) -> Dict[str, Any]:
        """Start streaming camera frames to a remote host.

        :param host: The IP or hostname the device should stream to.
        :param port: TCP port for the stream (default 9000).
        :param quality: Optional JPEG quality override (0-63, lower = higher fidelity).
        """
        host_value = (host or "").strip()
        if not host_value:
            raise ValueError("Host cannot be empty")

        if port is not None:
            if port <= 0 or port > 65535:
                raise ValueError("Port must be between 1 and 65535")
        if quality is not None:
            if quality < 0 or quality > 63:
                raise ValueError("Quality must be between 0 and 63 (lower is higher fidelity)")

        data = host_value
        if port is not None:
            data += f"|{port}"
        elif quality is not None:
            # Preserve position for quality when port is omitted
            data += "|"
        if quality is not None:
            data += f"|{quality}"

        return self._send_command("rdmp_stream", data)

    def stop_rdmp_stream(self) -> Dict[str, Any]:
        """Stop the remote streaming session."""
        return self._send_command("rdmp_stop")

    def stream_with_visualization(
        self,
        port: int = 9000,
        host: str = "",
        remote_host: Optional[str] = None,
        remote_port: int = 9000,
        hand_tracking: bool = False,
        quality: Optional[int] = None,
        window_name: str = "Voxel Stream",
        connect_timeout: float = 5.0,
    ) -> None:
        """Start streaming from the device and visualize frames locally.

        :param port: Local TCP port to listen on for incoming frames.
        :param host: Local interface to bind (default all interfaces).
        :param remote_host: Optional explicit remote address to push to. If
            None, the device is instructed to push to our current LAN IP.
        :param remote_port: Port the device should connect back to (defaults to
            the same as the local listener port).
        :param window_name: OpenCV window title.
        :param connect_timeout: Seconds to wait for the device to connect.
        :param quality: Optional JPEG quality override (0-63, lower numbers increase fidelity / bandwidth).
        :param hand_tracking: If True, overlay MediaPipe hand landmarks over each frame.
        """

        if not self.is_connected():
            raise ConnectionError("Connect to the device first")

        if cv2 is None or np is None:
            raise RuntimeError("OpenCV (cv2) and numpy are required for stream visualization. Install them with `pip install opencv-python numpy`." )

        hand_overlay = create_hand_overlay(hand_tracking)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)

        target_port = remote_port or port
        target_host = self._select_stream_target(remote_host, target_port)

        # Kick off streaming on device
        response = self.start_rdmp_stream(target_host, target_port, quality=quality)
        if "error" in response:
            listener.close()
            raise RuntimeError(f"Device failed to start streaming: {response}")

        listener.settimeout(connect_timeout)
        try:
            conn, addr = listener.accept()
        except socket.timeout:
            listener.close()
            self.stop_rdmp_stream()
            raise TimeoutError("Timed out waiting for stream connection from device")

        listener.close()
        conn.settimeout(5.0)

        print(f"Streaming from device connected: {addr}")

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            while True:
                header = self._recv_exact(conn, 8)
                if not header:
                    print("Stream closed by device")
                    break

                if header[:4] != b"VXL0":
                    print("Invalid frame header, stopping")
                    break

                frame_len = struct.unpack(">I", header[4:])[0]
                if frame_len <= 0 or frame_len > 5 * 1024 * 1024:
                    print(f"Invalid frame length: {frame_len}")
                    break

                payload = self._recv_exact(conn, frame_len)
                if not payload:
                    print("Failed to read frame payload")
                    break

                frame_array = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if image is None:
                    print("Failed to decode JPEG frame")
                    continue

                if hand_overlay:
                    hand_overlay.annotate(image)

                cv2.imshow(window_name, image)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    print("Stopping stream visualization")
                    break

        finally:
            conn.close()
            cv2.destroyWindow(window_name)
            if hand_overlay:
                hand_overlay.close()
            try:
                self.stop_rdmp_stream()
            except Exception:
                pass

    def _recv_exact(self, conn: socket.socket, length: int) -> bytes:
        remaining = length
        chunks = []
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                return b""
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _select_stream_target(self, remote_host: Optional[str], remote_port: int) -> str:
        if remote_host and remote_host.strip():
            return remote_host.strip()

        def _is_valid(ip: Optional[str]) -> bool:
            return bool(ip) and not ip.startswith("127.") and ip != "0.0.0.0"

        def _gather_local_ipv4_addresses() -> List[str]:
            addresses: Set[str] = set()
            hostname = socket.gethostname()
            try:
                infos = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_DGRAM)
                for info in infos:
                    addr = info[4][0]
                    if _is_valid(addr):
                        addresses.add(addr)
            except Exception:
                pass
            try:
                direct = socket.gethostbyname(hostname)
                if _is_valid(direct):
                    addresses.add(direct)
            except Exception:
                pass
            try:
                import psutil  # type: ignore

                for iface_addrs in psutil.net_if_addrs().values():
                    for iface in iface_addrs:
                        if getattr(iface, "family", None) == socket.AF_INET:
                            addr = getattr(iface, "address", None)
                            if _is_valid(addr):
                                addresses.add(addr)
            except Exception:
                pass
            return list(addresses)

        def _device_network() -> Optional[ipaddress.IPv4Network]:
            try:
                status = self.get_wifi_status()
            except Exception:
                return None
            if not isinstance(status, dict):
                return None
            if status.get("connected") is not True:
                return None
            ip_value = status.get("ip")
            subnet = status.get("subnet")
            if not ip_value or not subnet:
                return None
            try:
                return ipaddress.IPv4Network(f"{ip_value}/{subnet}", strict=False)
            except Exception:
                return None

        local_addresses = _gather_local_ipv4_addresses()
        network = _device_network()
        if network:
            for candidate in local_addresses:
                try:
                    if ipaddress.IPv4Address(candidate) in network:
                        return candidate
                except Exception:
                    continue

        # Fallback to previous heuristics if no matching interface was found
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as temp:
                temp.connect(("8.8.8.8", 80))
                candidate = temp.getsockname()[0]
                if _is_valid(candidate):
                    return candidate
        except Exception:
            pass

        try:
            hostname_ips = socket.gethostbyname_ex(socket.gethostname())[2]
            for ip in hostname_ips:
                if _is_valid(ip):
                    return ip
        except Exception:
            pass

        raise RuntimeError(
            "Unable to determine local IP address for streaming. "
            "Specify the host explicitly, e.g. stream 192.168.1.10 9000"
        )