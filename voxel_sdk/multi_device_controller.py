from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import socket
import struct
from queue import Queue, Empty

from .voxel import cv2, np, create_hand_overlay


@dataclass
class _ControllerEntry:
    label: str
    name: str
    controller: Any


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
                window = f"{window_name} ({entry.name})"
                connections[entry.label] = {
                    "entry": entry,
                    "conn": conn,
                    "addr": addr,
                    "filesystem": filesystem,
                    "queue": queue,
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
                        if hand_overlay:
                            hand_overlay.annotate(image)
                        try:
                            queue.put(image, timeout=0.1)
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

            # Step 3: Display loop on main thread
            while not stop_event.is_set():
                for info in connections.values():
                    queue = info["queue"]
                    try:
                        frame = queue.get(timeout=0.01)
                    except Empty:
                        continue
                    if frame is None:
                        stop_event.set()
                        break
                    cv2.imshow(info["window"], frame)

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


