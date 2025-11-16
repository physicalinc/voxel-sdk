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

from .voxel import cv2, np, create_hand_overlay


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


def _render_3d_hand(
    points_3d: List[Optional[Tuple[float, float, float]]],
    width: int = 640,
    height: int = 480,
) -> "np.ndarray":
    """Render 3D hand points in a 2D visualization window (XY view with Z as color)."""
    if np is None or cv2 is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create blank canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Filter valid points
    valid_points = [(i, p) for i, p in enumerate(points_3d) if p is not None]
    if not valid_points:
        return canvas
    
    # Extract coordinates
    coords = np.array([p for _, p in valid_points])
    
    # Normalize coordinates to fit in view
    if len(coords) > 0:
        # Center and scale
        center = coords.mean(axis=0)
        coords_centered = coords - center
        
        # Scale to fit (assuming reasonable range)
        max_dist = np.abs(coords_centered).max()
        if max_dist > 0:
            scale = min(width, height) * 0.35 / max_dist
            coords_scaled = coords_centered * scale
        
        # Project to 2D (XY view, Z as depth)
        origin_x = width // 2
        origin_y = height // 2
        
        # Draw axes
        axis_len = 50
        cv2.line(canvas, (origin_x, origin_y), (origin_x + axis_len, origin_y), (0, 0, 255), 2)  # X axis (red)
        cv2.line(canvas, (origin_x, origin_y), (origin_x, origin_y - axis_len), (0, 255, 0), 2)  # Y axis (green)
        cv2.putText(canvas, "X", (origin_x + axis_len + 5, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "Y", (origin_x, origin_y - axis_len - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, "Z (depth)", (origin_x - 60, origin_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw hand connections (MediaPipe hand connections)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]
        
        # Map indices to valid points
        idx_map = {valid_points[i][0]: i for i in range(len(valid_points))}
        
        for start_idx, end_idx in connections:
            if start_idx in idx_map and end_idx in idx_map:
                start_2d = coords_scaled[idx_map[start_idx]]
                end_2d = coords_scaled[idx_map[end_idx]]
                
                x1 = int(origin_x + start_2d[0])
                y1 = int(origin_y - start_2d[1])  # Flip Y for screen coordinates
                x2 = int(origin_x + end_2d[0])
                y2 = int(origin_y - end_2d[1])
                
                # Color by depth (Z) - closer = brighter
                z1 = coords[idx_map[start_idx]][2]
                z2 = coords[idx_map[end_idx]][2]
                avg_z = (z1 + z2) / 2.0
                # Normalize depth (assume 200-1000mm range)
                depth_norm = max(0.0, min(1.0, (avg_z - 200) / 800.0))
                depth_color = int(255 * (1.0 - depth_norm))
                depth_color = max(50, min(255, depth_color))
                
                cv2.line(canvas, (x1, y1), (x2, y2), (depth_color, depth_color, 255), 2)
        
        # Draw points
        for i, (orig_idx, coord) in enumerate(valid_points):
            pt_2d = coords_scaled[i]
            x = int(origin_x + pt_2d[0])
            y = int(origin_y - pt_2d[1])
            z = coord[2]
            
            # Color by depth
            depth_norm = max(0.0, min(1.0, (z - 200) / 800.0))
            depth_color = int(255 * (1.0 - depth_norm))
            depth_color = max(50, min(255, depth_color))
            
            cv2.circle(canvas, (x, y), 5, (depth_color, depth_color, 255), -1)
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)
            
            # Label key points
            if orig_idx in [4, 8, 12, 16, 20]:  # Finger tips
                labels = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
                label = labels.get(orig_idx, "")
                cv2.putText(canvas, label, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show depth info
        if len(valid_points) > 8 and points_3d[8] is not None:
            idx_tip = points_3d[8]
            info_text = f"Index Tip: Z={idx_tip[2]:.1f}mm"
            cv2.putText(canvas, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return canvas


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
            if not has_all_calibrations:
                print("⚠️  Warning: Calibration data not found for both cameras. 3D triangulation disabled.")
                print("   Run 'calibrate' on both devices to enable 3D hand tracking.")
            else:
                print("✓ Calibration data loaded for both cameras. 3D hand tracking enabled.")

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

                conn.settimeout(5.0)
                queue: Queue[Any] = Queue(maxsize=2)
                landmarks_queue: Queue[Any] = Queue(maxsize=2) if hand_tracking else None
                frame_buffer: deque = deque(maxsize=5) if (hand_tracking and len(self._controllers) == 2) else None
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
                data = bytearray()
                while len(data) < length and not stop_event.is_set():
                    try:
                        chunk = sock.recv(length - len(data))
                    except socket.timeout:
                        continue
                    if not chunk:
                        return None
                    data.extend(chunk)
                if stop_event.is_set():
                    return None
                return bytes(data)

            def reader(device_label: str, info: Dict[str, Any]) -> None:
                conn = info["conn"]
                queue = info["queue"]
                landmarks_queue = info.get("landmarks_queue")
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
                        
                        try:
                            queue.put(image, timeout=0.1)
                            if landmarks_queue is not None:
                                landmarks_queue.put(landmarks_data, timeout=0.1)
                            # Store timestamped frame+landmarks for synchronization
                            if frame_buffer is not None:
                                frame_buffer.append((time.time(), image, landmarks_data))
                        except Exception:
                            pass
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
            
            # Create 3D visualization window if hand tracking enabled
            if hand_tracking and len(self._controllers) == 2:
                cv2.namedWindow("3D Hand Visualization", cv2.WINDOW_NORMAL)

            # Step 3: Display loop on main thread with stereo triangulation
            while not stop_event.is_set():
                frames: Dict[str, Any] = {}
                landmarks_dict: Dict[str, Any] = {}
                
                # Collect synchronized frames and landmarks from all devices
                if hand_tracking and len(self._controllers) == 2 and all(info.get("frame_buffer") for info in connections.values()):
                    # Use frame buffers for synchronization
                    synced_data: Dict[str, Tuple[float, Any, Any]] = {}
                    for label, info in connections.items():
                        frame_buffer = info.get("frame_buffer")
                        if frame_buffer and len(frame_buffer) > 0:
                            # Get most recent frame
                            synced_data[label] = frame_buffer[-1]
                    
                    # If we have both frames, check if they're close in time (within 50ms)
                    if len(synced_data) == 2 and "left" in synced_data and "right" in synced_data:
                        left_time, left_frame, left_landmarks = synced_data["left"]
                        right_time, right_frame, right_landmarks = synced_data["right"]
                        time_diff = abs(left_time - right_time)
                        
                        if time_diff < 0.05:  # Within 50ms - good sync
                            frames["left"] = left_frame
                            frames["right"] = right_frame
                            landmarks_dict["left"] = left_landmarks
                            landmarks_dict["right"] = right_landmarks
                        else:
                            # Try to find better matching frames
                            best_match_diff = float('inf')
                            best_left = None
                            best_right = None
                            
                            left_buffer = connections["left"].get("frame_buffer", deque())
                            right_buffer = connections["right"].get("frame_buffer", deque())
                            
                            for left_item in left_buffer:
                                for right_item in right_buffer:
                                    diff = abs(left_item[0] - right_item[0])
                                    if diff < best_match_diff and diff < 0.1:  # Within 100ms
                                        best_match_diff = diff
                                        best_left = left_item
                                        best_right = right_item
                            
                            if best_left and best_right:
                                frames["left"] = best_left[1]
                                frames["right"] = best_right[1]
                                landmarks_dict["left"] = best_left[2]
                                landmarks_dict["right"] = best_right[2]
                            else:
                                # Fall back to most recent
                                frames["left"] = left_frame
                                frames["right"] = right_frame
                                landmarks_dict["left"] = left_landmarks
                                landmarks_dict["right"] = right_landmarks
                    else:
                        # Fall back to regular queue method
                        for label, info in connections.items():
                            queue = info["queue"]
                            try:
                                frame = queue.get(timeout=0.01)
                                if frame is None:
                                    stop_event.set()
                                    break
                                frames[label] = frame
                                
                                if info.get("landmarks_queue"):
                                    try:
                                        landmarks = info["landmarks_queue"].get(timeout=0.01)
                                        landmarks_dict[label] = landmarks
                                    except Empty:
                                        landmarks_dict[label] = None
                            except Empty:
                                continue
                else:
                    # Regular collection for single device or no hand tracking
                    for label, info in connections.items():
                        queue = info["queue"]
                        try:
                            frame = queue.get(timeout=0.01)
                            if frame is None:
                                stop_event.set()
                                break
                            frames[label] = frame
                            
                            if hand_tracking and info.get("landmarks_queue"):
                                try:
                                    landmarks = info["landmarks_queue"].get(timeout=0.01)
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
                            
                            # Render 3D visualization
                            valid_3d_points = [p for p in points_3d if p is not None]
                            if len(valid_3d_points) > 0:
                                viz_3d = _render_3d_hand(points_3d)
                                cv2.imshow("3D Hand Visualization", viz_3d)
                                
                                # Display coordinates on frames (index finger tip is landmark 8)
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
                
                # Display all frames
                for label, frame in frames.items():
                    cv2.imshow(connections[label]["window"], frame)

                key = cv2.waitKey(1)
                if key in (27, ord("q")):
                    stop_event.set()
                    break

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


