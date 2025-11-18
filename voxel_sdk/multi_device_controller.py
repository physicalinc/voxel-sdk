from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import socket
import struct
from queue import Queue, Empty

from voxel_sdk.voxel import cv2, np, create_hand_overlay


@dataclass
class _ControllerEntry:
    label: str
    name: str
    controller: Any


def _load_calibration(controller: Any, label: str) -> Optional[Dict[str, Any]]:
    """Load calibration data for a device."""
    try:
        filename = f"intrinsics_{label}.json" if label in ("left", "right") else "intrinsics.json"
        calib_path = f"/calibration/{filename}"
        exists_resp = controller.execute_device_command(f"file_exists:{calib_path}")
        if isinstance(exists_resp, dict) and exists_resp.get("exists"):
            read_resp = controller.execute_device_command(f"read_file:{calib_path}")
            if isinstance(read_resp, dict) and "content" in read_resp:
                return json.loads(read_resp["content"])
    except Exception:
        pass
    return None


def _stereo_triangulate(
    point_left: Tuple[float, float],
    point_right: Tuple[float, float],
    K_left: "np.ndarray",
    K_right: "np.ndarray",
    baseline_mm: float = 65.0,  # Default interpupillary distance in mm
) -> Optional[Tuple[float, float, float]]:
    """Triangulate 3D point from 2D correspondences in stereo cameras.
    
    Assumes parallel cameras with known baseline.
    Returns (X, Y, Z) in mm relative to left camera.
    """
    if np is None or cv2 is None:
        return None
    
    try:
        xl, yl = point_left
        xr, yr = point_right
        
        # Get focal lengths (assuming fx ≈ fy)
        fx_left = float(K_left[0, 0])
        fx_right = float(K_right[0, 0])
        
        # Disparity: for parallel cameras, left camera sees point more to the right
        # So xl should be > xr for objects in front, making disparity positive
        disparity = xl - xr
        
        # Check if disparity is reasonable (at least 1 pixel)
        if abs(disparity) < 1.0:  # Too small disparity, unreliable
            return None
        
        # Depth from disparity: Z = (baseline * focal_length) / disparity
        # Use average focal length
        fx_avg = (fx_left + fx_right) / 2.0
        Z = (baseline_mm * fx_avg) / abs(disparity)
        
        # Sanity check: Z should be positive and reasonable (between 10mm and 5000mm)
        if Z <= 0 or Z > 5000:
            return None
        
        # X and Y coordinates (using left camera)
        cx_left = float(K_left[0, 2])
        cy_left = float(K_left[1, 2])
        
        X = ((xl - cx_left) * Z) / fx_left
        Y = ((yl - cy_left) * Z) / fx_left
        
        return (X, Y, Z)
    except Exception:
        return None


class MultiDeviceController:
    """Aggregate multiple DeviceController instances and broadcast commands."""

    is_multi = True

    def __init__(self, controllers: Dict[str, Any], display_names: Optional[Dict[str, str]] = None):
        if not controllers:
            raise ValueError("At least one controller is required")

        self._controllers: Dict[str, _ControllerEntry] = {}
        for label, controller in controllers.items():
            if not hasattr(controller, "execute_device_command"):
                raise TypeError(f"Controller '{label}' does not implement execute_device_command")
            name = display_names.get(label, label) if display_names else label
            self._controllers[label] = _ControllerEntry(label=label, name=name, controller=controller)

    # --- Properties ---
    @property
    def device_labels(self) -> Dict[str, str]:
        """Mapping of logical label -> user-friendly name."""
        return {label: entry.name for label, entry in self._controllers.items()}

    @property
    def controllers(self) -> Dict[str, Any]:
        """Access underlying controllers by label (e.g., 'left', 'right')."""
        return {label: entry.controller for label, entry in self._controllers.items()}

    def get_controller(self, label: str) -> Optional[Any]:
        """Return a single underlying controller by label, or None if not present."""
        entry = self._controllers.get(label)
        return entry.controller if entry else None

    # --- Helpers ---
    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _broadcast_command(self, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        responses: Dict[str, Any] = {}
        timing: Dict[str, Dict[str, Any]] = {}
        threads = []
        lock = threading.Lock()
        start_wall = self._utc_now_iso()

        def invoke(entry: _ControllerEntry) -> None:
            send_ts = self._utc_now_iso()
            send_perf = time.perf_counter()
            try:
                func = getattr(entry.controller, func_name)
                result = func(*args, **kwargs)
                error = None
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
                error = str(exc)
            recv_perf = time.perf_counter()
            recv_ts = self._utc_now_iso()

            with lock:
                responses[entry.label] = result
                timing[entry.label] = {
                    "device_name": entry.name,
                    "sent_at": send_ts,
                    "received_at": recv_ts,
                    "latency_ms": round((recv_perf - send_perf) * 1000, 3),
                }
                if error is not None:
                    timing[entry.label]["error"] = error

        for entry in self._controllers.values():
            thread = threading.Thread(target=invoke, args=(entry,), daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return {
            "dispatched_at": start_wall,
            "time_sync": timing,
            "responses": responses,
        }

    # --- Public API ---
    def execute_device_command(self, device_command: str) -> Dict[str, Any]:
        return {
            "command": device_command,
            **self._broadcast_command("execute_device_command", device_command),
        }

    def stop_stream(self) -> Dict[str, Any]:
        return {
            "command": "rdmp_stop",
            **self._broadcast_command("stop_stream"),
        }

    def disconnect(self) -> None:
        for entry in self._controllers.values():
            try:
                entry.controller.disconnect()
            except Exception:
                pass  # Ignore disconnect errors for shutdown

    # Streaming support for multiple devices
    def stream_with_visualization(
        self,
        port: int = 9000,
        host: str = "",
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        hand_tracking: bool = False,
        quality: Optional[int] = None,
        window_name: str = "Voxel Stream",
        connect_timeout: float = 5.0,
    ) -> None:
        """Launch concurrent stream viewers for all connected devices."""
        if cv2 is None or np is None:
            raise RuntimeError(
                "OpenCV (cv2) and numpy are required for stream visualization. "
                "Install them with `pip install opencv-python numpy`."
            )
        if hand_tracking:
            # quick dependency check
            tester = create_hand_overlay(True)
            tester.close()
        if cv2 is None or np is None:
            raise RuntimeError(
                "OpenCV (cv2) and numpy are required for stream visualization. "
                "Install them with `pip install opencv-python numpy`."
            )

        base_remote_port = remote_port if remote_port is not None else port
        listeners = []
        connections: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        stop_event = threading.Event()

        # Load calibration data for stereo triangulation
        calibrations: Dict[str, Optional[Dict[str, Any]]] = {}
        K_matrices: Dict[str, Optional["np.ndarray"]] = {}
        if hand_tracking and len(self._controllers) == 2:
            for label, entry in self._controllers.items():
                calib = _load_calibration(entry.controller, label)
                calibrations[label] = calib
                if calib and np is not None:
                    K_matrices[label] = np.array(calib.get("camera_matrix", []))
                else:
                    K_matrices[label] = None
            has_all_calibrations = all(K is not None for K in K_matrices.values())
            if has_all_calibrations:
                print("✓ Calibration data loaded for both cameras. 3D computation enabled.")

        try:
            # Step 1: Prepare listeners and start device streams
            for index, entry in enumerate(self._controllers.values()):
                local_port = port + index
                target_port = base_remote_port + index

                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((host, local_port))
                listener.listen(1)
                listeners.append(listener)

                filesystem = entry.controller.filesystem  # type: ignore[attr-defined]
                target_host = filesystem._select_stream_target(remote_host, target_port)  # noqa: SLF001
                response = filesystem.start_rdmp_stream(target_host, target_port, quality=quality)
                if isinstance(response, dict) and "error" in response:
                    raise RuntimeError(f"{entry.label} failed to start streaming: {response}")

                listener.settimeout(connect_timeout)
                try:
                    conn, addr = listener.accept()
                except socket.timeout:
                    filesystem.stop_rdmp_stream()
                    raise TimeoutError(f"{entry.label} timed out waiting for stream connection")
                finally:
                    listener.close()

                conn.settimeout(None)  # Blocking reads - let TCP handle buffering
                # Larger queues to prevent blocking - allow up to 10 frames per stream
                queue: Queue[Any] = Queue(maxsize=10)
                landmarks_queue: Queue[Any] = Queue(maxsize=10) if hand_tracking else None
                frame_buffer: deque = deque(maxlen=10) if (hand_tracking and len(self._controllers) == 2) else None
                window = f"{window_name} ({entry.name})"
                connections[entry.label] = {
                    "entry": entry,
                    "conn": conn,
                    "addr": addr,
                    "filesystem": filesystem,
                    "queue": queue,
                    "landmarks_queue": landmarks_queue,
                    "frame_buffer": frame_buffer,
                    "window": window,
                }
                print(f"{entry.label} stream connected: {addr}")

            # Step 2: Launch reader threads (decode in background, display in main thread)
            def recv_exact(sock: socket.socket, length: int) -> Optional[bytes]:
                """Efficiently receive exact number of bytes without timeout delays."""
                data = bytearray()
                while len(data) < length and not stop_event.is_set():
                    try:
                        chunk = sock.recv(length - len(data))
                        if not chunk:
                            return None
                        data.extend(chunk)
                    except socket.timeout:
                        # Should not happen with blocking socket, but handle gracefully
                        if stop_event.is_set():
                            return None
                        continue
                    except OSError:
                        # Connection closed or error
                        return None
                if stop_event.is_set():
                    return None
                return bytes(data)

            def reader(device_label: str, info: Dict[str, Any]) -> None:
                conn = info["conn"]
                queue = info["queue"]
                landmarks_queue = info.get("landmarks_queue")
                frame_buffer = info.get("frame_buffer")
                filesystem = info["filesystem"]
                entry = info["entry"]
                hand_overlay = None
                try:
                    hand_overlay = create_hand_overlay(hand_tracking)
                    magic = b"VXL0"
                    while not stop_event.is_set():
                        header = recv_exact(conn, 8)
                        if not header:
                            break
                        if header[:4] != magic:
                            errors[device_label] = "Invalid frame header"
                            break
                        frame_len = struct.unpack(">I", header[4:])[0]
                        if frame_len <= 0 or frame_len > 5 * 1024 * 1024:
                            errors[device_label] = f"Invalid frame length: {frame_len}"
                            break
                        payload = recv_exact(conn, frame_len)
                        if not payload:
                            break
                        frame_array = np.frombuffer(payload, dtype=np.uint8)
                        image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        if image is None:
                            continue
                        
                        landmarks_data = None
                        if hand_overlay:
                            # Process first to get landmarks, then annotate
                            if landmarks_queue is not None:
                                landmarks_data = hand_overlay.process(image)
                            hand_overlay.annotate(image)
                        
                        # Non-blocking put - drop frame if queue is full to prevent slowdown
                        # This ensures reader threads never block waiting for main thread
                        try:
                            queue.put_nowait(image)
                        except Exception:
                            # Queue full - drop this frame to maintain throughput
                            pass
                        
                        if landmarks_queue is not None:
                            try:
                                landmarks_queue.put_nowait(landmarks_data)
                            except Exception:
                                pass
                        
                        # Store timestamped frame+landmarks for synchronization
                        if frame_buffer is not None:
                            frame_buffer.append((time.time(), image, landmarks_data))
                except Exception as exc:  # noqa: BLE001
                    errors[device_label] = str(exc)
                finally:
                    stop_event.set()
                    try:
                        queue.put(None, timeout=0.1)
                    except Exception:
                        pass
                    try:
                        filesystem.stop_rdmp_stream()
                    except Exception:
                        pass
                    if hand_overlay:
                        hand_overlay.close()
                    conn.close()

            threads = []
            for device_label, info in connections.items():
                thread = threading.Thread(target=reader, args=(device_label, info), daemon=True)
                threads.append(thread)
                thread.start()

            for info in connections.values():
                cv2.namedWindow(info["window"], cv2.WINDOW_NORMAL)

            # Step 3: Display loop on main thread with stereo triangulation
            while not stop_event.is_set():
                frames: Dict[str, Any] = {}
                landmarks_dict: Dict[str, Any] = {}
                
                # Collect synchronized frames and landmarks from all devices
                # Optimized: Use queues directly for better performance, only use buffers for sync when needed
                if hand_tracking and len(self._controllers) == 2 and all(info.get("frame_buffer") for info in connections.values()):
                    # Fast path: Try to get frames from queues first (most recent)
                    for label, info in connections.items():
                        queue = info["queue"]
                        try:
                            # Get the most recent frame (drain queue to get latest)
                            frame = None
                            while True:
                                try:
                                    frame = queue.get_nowait()
                                    if frame is None:
                                        stop_event.set()
                                        break
                                except Empty:
                                    break
                            if frame is not None:
                                frames[label] = frame
                                
                                if info.get("landmarks_queue"):
                                    landmarks = None
                                    try:
                                        # Get most recent landmarks
                                        while True:
                                            try:
                                                landmarks = info["landmarks_queue"].get_nowait()
                                            except Empty:
                                                break
                                        landmarks_dict[label] = landmarks
                                    except Empty:
                                        landmarks_dict[label] = None
                        except Empty:
                            pass
                    
                    # If we have both frames, use frame buffers for synchronization only if needed
                    if len(frames) == 2 and "left" in frames and "right" in frames:
                        # Check if frames from buffers are better synchronized
                        left_buffer = connections["left"].get("frame_buffer", deque())
                        right_buffer = connections["right"].get("frame_buffer", deque())
                        
                        if len(left_buffer) > 0 and len(right_buffer) > 0:
                            # Quick check: use most recent frames if they're close enough
                            left_latest = left_buffer[-1]
                            right_latest = right_buffer[-1]
                            time_diff = abs(left_latest[0] - right_latest[0])
                            
                            if time_diff < 0.1:  # Within 100ms - use buffer frames for better sync
                                frames["left"] = left_latest[1]
                                frames["right"] = right_latest[1]
                                landmarks_dict["left"] = left_latest[2]
                                landmarks_dict["right"] = right_latest[2]
                else:
                    # Regular collection for single device or no hand tracking - optimized
                    for label, info in connections.items():
                        queue = info["queue"]
                        try:
                            # Get most recent frame (drain to get latest)
                            frame = None
                            while True:
                                try:
                                    frame = queue.get_nowait()
                                    if frame is None:
                                        stop_event.set()
                                        break
                                except Empty:
                                    break
                            
                            if frame is not None:
                                frames[label] = frame
                                
                                if hand_tracking and info.get("landmarks_queue"):
                                    landmarks = None
                                    try:
                                        # Get most recent landmarks
                                        while True:
                                            try:
                                                landmarks = info["landmarks_queue"].get_nowait()
                                            except Empty:
                                                break
                                        landmarks_dict[label] = landmarks
                                    except Empty:
                                        landmarks_dict[label] = None
                        except Empty:
                            continue
                
                if stop_event.is_set():
                    break
                
                # Perform stereo triangulation if we have both cameras and calibrations
                has_all_calibrations = all(K is not None for K in K_matrices.values())
                if (hand_tracking and len(frames) == 2 and 
                    "left" in frames and "right" in frames and
                    has_all_calibrations):
                    
                    left_landmarks = landmarks_dict.get("left")
                    right_landmarks = landmarks_dict.get("right")
                    K_left = K_matrices["left"]
                    K_right = K_matrices["right"]
                    
                    if left_landmarks and right_landmarks and K_left is not None and K_right is not None:
                        # Match and triangulate corresponding landmarks
                        # For simplicity, match first hand from each camera
                        if len(left_landmarks) > 0 and len(right_landmarks) > 0:
                            left_hand = left_landmarks[0]  # 21 landmarks
                            right_hand = right_landmarks[0]  # 21 landmarks
                            
                            # Triangulate each corresponding landmark point
                            points_3d: List[Optional[Tuple[float, float, float]]] = []
                            for i in range(min(len(left_hand), len(right_hand))):
                                pt_left = (float(left_hand[i][0]), float(left_hand[i][1]))
                                pt_right = (float(right_hand[i][0]), float(right_hand[i][1]))
                                pt_3d = _stereo_triangulate(pt_left, pt_right, K_left, K_right)
                                points_3d.append(pt_3d)
                            
                            # Display coordinates on frames (index finger tip is landmark 8)
                            valid_3d_points = [p for p in points_3d if p is not None]
                            if len(valid_3d_points) > 0:
                                if len(valid_3d_points) > 8 and points_3d[8] is not None:
                                    idx_tip = points_3d[8]  # Index finger tip
                                    x, y, z = idx_tip
                                    text = f"3D: X={x:.1f}mm Y={y:.1f}mm Z={z:.1f}mm"
                                    cv2.putText(
                                        frames["left"],
                                        text,
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )
                                    
                                    # Also show on right frame
                                    cv2.putText(
                                        frames["right"],
                                        text,
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )
                                else:
                                    # Show status if some points triangulated but not index tip
                                    status_text = f"3D tracking: {len(valid_3d_points)}/21 points"
                                    cv2.putText(
                                        frames["left"],
                                        status_text,
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 255),
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    cv2.putText(
                                        frames["right"],
                                        status_text,
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 255),
                                        1,
                                        cv2.LINE_AA,
                                    )
                
                # Display all frames (only if we have frames to show)
                if frames:
                    for label, frame in frames.items():
                        cv2.imshow(connections[label]["window"], frame)
                    
                    key = cv2.waitKey(1)
                    if key in (27, ord("q")):
                        stop_event.set()
                        break
                else:
                    # No frames available - small delay to prevent CPU spinning
                    time.sleep(0.001)

            for thread in threads:
                thread.join(timeout=1.0)

            for info in connections.values():
                cv2.destroyWindow(info["window"])

            if errors:
                raise RuntimeError(f"Streaming failed for: {errors}")

        finally:
            stop_event.set()
            for listener in listeners:
                try:
                    listener.close()
                except Exception:
                    pass
            for info in connections.values():
                try:
                    info["conn"].close()
                except Exception:
                    pass
                try:
                    cv2.destroyWindow(info["window"])
                except Exception:
                    pass
                try:
                    info["filesystem"].stop_rdmp_stream()
                except Exception:
                    pass

    def download_file(self, *args, **kwargs):
        raise NotImplementedError("Downloading files is not supported in multi-device mode")

    def download_video(self, *args, **kwargs):
        raise NotImplementedError("Downloading videos is not supported in multi-device mode")


