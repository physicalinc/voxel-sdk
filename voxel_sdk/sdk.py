from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from voxel_sdk.calibration import CalibrationResult, calibrate_and_save
from voxel_sdk.commands import ParsedCommand, generate_help_text, parse_command
from voxel_sdk.device_controller import DeviceController, DownloadSummary
from voxel_sdk.multi_device_controller import MultiDeviceController


ControllerLike = Union[DeviceController, MultiDeviceController]
ConfirmCallback = Optional[Callable[[str, Optional[Dict[str, Any]]], bool]]


class TerminalCommandError(RuntimeError):
    """Raised when a terminal command cannot be executed via the SDK."""


@dataclass
class FileDownloadResult:
    device_path: str
    size_bytes: int
    saved_to: Optional[str] = None
    data: Optional[bytes] = None
    side: Optional[str] = None


class TerminalAPI:
    """Programmatic interface that mirrors the Voxel terminal commands."""

    def __init__(self, controller: ControllerLike):
        self._controller = controller

    @property
    def controller(self) -> ControllerLike:
        return self._controller

    @property
    def is_multi(self) -> bool:
        return bool(getattr(self._controller, "is_multi", False))

    def help(self) -> str:
        """Return the same help text printed by the CLI."""
        return generate_help_text()

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    def _ensure_leading_slash(self, path: str) -> str:
        if not path:
            return "/"
        return path if path.startswith("/") else f"/{path}"

    def _multi(self) -> MultiDeviceController:
        if not self.is_multi:
            raise TerminalCommandError("Operation requires a MultiDeviceController")
        return self._controller  # type: ignore[return-value]

    def _get_controller(self, side: Optional[str], *, required: bool = True) -> DeviceController:
        if self.is_multi:
            if side:
                controller = self._multi().get_controller(side)
                if controller is None:
                    raise TerminalCommandError(f"No '{side}' device is connected.")
                return controller
            if required:
                raise TerminalCommandError(
                    "Specify a device side when using a MultiDeviceController (e.g. side='left')."
                )
            raise TerminalCommandError("Device side required for this operation.")
        assert isinstance(self._controller, DeviceController)
        return self._controller

    def _ordered_labels(self) -> List[str]:
        if not self.is_multi:
            return []
        labels = list(self._multi().device_labels.keys())
        ordered: List[str] = []
        for preferred in ("left", "right"):
            if preferred in labels:
                ordered.append(preferred)
        for label in labels:
            if label not in ordered:
                ordered.append(label)
        return ordered

    def _calibration_path(self, label: Optional[str]) -> str:
        if label in ("left", "right"):
            filename = f"intrinsics_{label}.json"
        else:
            filename = "intrinsics.json"
        return f"/calibration/{filename}"

    def _fetch_calibration(self, controller: DeviceController, label: Optional[str]) -> Optional[Dict[str, Any]]:
        path = self._calibration_path(label)
        try:
            exists = controller.execute_device_command(f"file_exists:{path}")
            if not isinstance(exists, dict) or not exists.get("exists"):
                return None
            payload = controller.execute_device_command(f"read_file:{path}")
        except Exception as exc:  # pragma: no cover - transport errors
            raise TerminalCommandError(f"Failed to read calibration data: {exc}") from exc
        if not isinstance(payload, dict) or "content" not in payload:
            return None
        try:
            return json.loads(payload["content"])
        except Exception as exc:
            raise TerminalCommandError(f"Invalid calibration data at {path}: {exc}") from exc

    def _build_device_command(self, name: str, payload: Optional[str] = None) -> str:
        if payload is None:
            return name
        return f"{name}:{payload}"

    def execute_device_command(self, device_command: str, *, side: Optional[str] = None) -> Any:
        """Send a raw device command string to the controller."""
        try:
            if self.is_multi:
                if side:
                    controller = self._get_controller(side)
                    return controller.execute_device_command(device_command)
                return self._multi().execute_device_command(device_command)
            return self._controller.execute_device_command(device_command)  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - transport errors
            raise TerminalCommandError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Filesystem commands
    # ------------------------------------------------------------------ #
    def card_info(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("card_info", side=side)

    def list_dir(self, path: str = "/", *, side: Optional[str] = None) -> Any:
        target = self._ensure_leading_slash(path)
        return self.execute_device_command(self._build_device_command("list_dir", target), side=side)

    def read_file(self, path: str, *, side: Optional[str] = None) -> Any:
        target = self._ensure_leading_slash(path)
        return self.execute_device_command(self._build_device_command("read_file", target), side=side)

    def write_file(self, path: str, content: str, *, side: Optional[str] = None) -> Any:
        target = self._ensure_leading_slash(path)
        payload = f"{target}|{content}"
        return self.execute_device_command(self._build_device_command("write_file", payload), side=side)

    def append_file(self, path: str, content: str, *, side: Optional[str] = None) -> Any:
        target = self._ensure_leading_slash(path)
        payload = f"{target}|{content}"
        return self.execute_device_command(self._build_device_command("append_file", payload), side=side)

    def delete_file(self, path: str, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command(self._build_device_command("delete_file", path), side=side)

    def file_exists(self, path: str, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command(self._build_device_command("file_exists", path), side=side)

    def file_size(self, path: str, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command(self._build_device_command("file_size", path), side=side)

    def create_dir(self, path: str, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command(self._build_device_command("create_dir", path), side=side)

    def remove_dir(self, path: str, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command(self._build_device_command("remove_dir", path), side=side)

    def rename_file(self, old_path: str, new_path: str, *, side: Optional[str] = None) -> Any:
        payload = f"{old_path},{new_path}"
        return self.execute_device_command(self._build_device_command("rename_file", payload), side=side)

    def download_file(
        self,
        device_path: str,
        *,
        side: Optional[str] = None,
        local_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        return_bytes: bool = True,
    ) -> FileDownloadResult:
        controller = self._get_controller(side)
        try:
            data = controller.download_file(device_path, progress_callback=progress_callback)
        except Exception as exc:  # pragma: no cover - transport errors
            raise TerminalCommandError(f"Download failed: {exc}") from exc

        saved_to: Optional[str] = None
        if local_filename:
            target = os.path.abspath(local_filename)
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(data)
            saved_to = target

        payload = data if return_bytes else None
        return FileDownloadResult(
            device_path=device_path,
            size_bytes=len(data),
            saved_to=saved_to,
            data=payload,
            side=side,
        )

    def download_video(
        self,
        video_dir: str,
        *,
        side: Optional[str] = None,
        output: Optional[str] = None,
        cleanup_frames: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> DownloadSummary:
        controller = self._get_controller(side)
        try:
            return controller.download_video(
                video_dir=video_dir,
                output=output,
                cleanup_frames=cleanup_frames,
                progress_callback=progress_callback,
            )
        except Exception as exc:  # pragma: no cover - transport errors
            raise TerminalCommandError(f"Video download failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Connectivity commands
    # ------------------------------------------------------------------ #
    def connect_wifi(self, ssid: str, password: str = "", *, side: Optional[str] = None) -> Any:
        if not ssid:
            raise TerminalCommandError("SSID cannot be empty.")
        payload = f"{ssid}|{password}"
        return self.execute_device_command(self._build_device_command("connectWifi", payload), side=side)

    def disconnect_wifi(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("disconnectWifi", side=side)

    def scan_wifi(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("scanWifi", side=side)

    def wifi_status(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("wifiStatus", side=side)

    def ping_host(self, host: str, count: int = 4, *, side: Optional[str] = None) -> Any:
        if not host:
            raise TerminalCommandError("Host cannot be empty.")
        count = max(1, min(count, 10))
        payload = f"{host}|{count}" if count else host
        return self.execute_device_command(self._build_device_command("ping_host", payload), side=side)

    # ------------------------------------------------------------------ #
    # Camera commands
    # ------------------------------------------------------------------ #
    def camera_status(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("camera_status", side=side)

    def camera_capture(
        self,
        directory: str = "/",
        name: str = "",
        resolution: str = "1600x1200",
        *,
        side: Optional[str] = None,
    ) -> Any:
        directory = directory or "/"
        payload = f"{directory}|{name}|{resolution}"
        return self.execute_device_command(self._build_device_command("camera_capture", payload), side=side)

    def camera_record(
        self,
        directory: str = "/",
        name: str = "",
        resolution: str = "800x600",
        fps: str = "30",
        *,
        side: Optional[str] = None,
    ) -> Any:
        payload = f"{directory}|{name}|{resolution}|{fps}"
        return self.execute_device_command(self._build_device_command("camera_record", payload), side=side)

    def camera_stop(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("camera_stop", side=side)

    def camera_config(
        self,
        resolution: str = "1600x1200",
        quality: str = "12",
        fmt: str = "JPEG",
        fb_count: str = "1",
        *,
        side: Optional[str] = None,
    ) -> Any:
        payload = f"{resolution}|{quality}|{fmt}|{fb_count}"
        return self.execute_device_command(self._build_device_command("camera_config", payload), side=side)

    def camera_reset(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("camera_reset", side=side)

    # ------------------------------------------------------------------ #
    # IMU commands
    # ------------------------------------------------------------------ #
    def imu_capture(
        self,
        *,
        save: bool = False,
        directory: str = "/",
        name: str = "",
        side: Optional[str] = None,
    ) -> Any:
        if save:
            directory = directory or "/"
            payload = f"--save|{directory}"
            if name:
                payload += f"|{name}"
            device_command = self._build_device_command("imu_capture", payload)
        else:
            device_command = "imu_capture"
        return self.execute_device_command(device_command, side=side)

    def imu_status(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("imu_status", side=side)

    def imu_stream_start(self, host: str, port: int = 9001, *, side: Optional[str] = None) -> Any:
        if not host:
            raise TerminalCommandError("Host cannot be empty.")
        payload = f"{host}|{port}" if port else host
        return self.execute_device_command(self._build_device_command("imu_stream", payload), side=side)

    def imu_stream_stop(self, *, side: Optional[str] = None) -> Any:
        return self.execute_device_command("imu_stop", side=side)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def stream(
        self,
        port: int = 9000,
        *,
        side: Optional[str] = None,
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        quality: Optional[int] = None,
        hand_tracking: bool = False,
        host: str = "",
        window_name: str = "Voxel Stream",
        connect_timeout: float = 5.0,
    ) -> None:
        if self.is_multi and side is None:
            multi = self._multi()
            multi.stream_with_visualization(
                port=port,
                host=host,
                remote_host=remote_host,
                remote_port=remote_port,
                quality=quality,
                hand_tracking=hand_tracking,
                window_name=window_name,
                connect_timeout=connect_timeout,
            )
            return

        controller = self._get_controller(side)
        controller.stream_with_visualization(
            port=port,
            host=host,
            remote_host=remote_host,
            remote_port=remote_port or port,
            quality=quality,
            hand_tracking=hand_tracking,
            window_name=window_name,
            connect_timeout=connect_timeout,
        )

    def stream_stop(self, *, side: Optional[str] = None) -> Any:
        if self.is_multi and side is None:
            return self._multi().stop_stream()
        controller = self._get_controller(side)
        return controller.stop_stream()

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        *,
        side: Optional[str] = None,
        samples: int = 0,
        force: bool = False,
        start_with_preview: bool = True,
        confirm_callback: ConfirmCallback = None,
    ) -> Union[CalibrationResult, Dict[str, CalibrationResult], None]:
        def should_proceed(label: Optional[str], existing: Optional[Dict[str, Any]]) -> bool:
            if existing is None:
                return True
            if force:
                return True
            if confirm_callback is not None:
                return bool(confirm_callback(label or "", existing))
            raise TerminalCommandError(
                f"Calibration already exists for '{label or 'device'}'. "
                "Pass force=True or provide confirm_callback to override."
            )

        if self.is_multi:
            multi = self._multi()
            targets: List[str]
            if side:
                targets = [side]
            else:
                targets = self._ordered_labels()
            results: Dict[str, CalibrationResult] = {}
            for label in targets:
                controller = multi.get_controller(label)
                if controller is None:
                    continue
                existing = self._fetch_calibration(controller, label)
                if not should_proceed(label, existing):
                    continue
                result = calibrate_and_save(
                    controller,
                    label=label,
                    requested_samples=samples,
                    start_with_preview=start_with_preview,
                )
                results[label] = result
            return results

        controller = self._get_controller(side=None)
        existing = self._fetch_calibration(controller, None)
        if not should_proceed(None, existing):
            return None
        return calibrate_and_save(
            controller,
            requested_samples=samples,
            start_with_preview=start_with_preview,
        )

    def calibration_info(self, *, side: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]], None]:
        if self.is_multi:
            multi = self._multi()
            targets = [side] if side else self._ordered_labels()
            result: Dict[str, Dict[str, Any]] = {}
            for label in targets:
                controller = multi.get_controller(label)
                if controller is None:
                    continue
                data = self._fetch_calibration(controller, label)
                if data is None:
                    result[label] = {"error": "Calibration not found"}
                else:
                    result[label] = data
            return result

        controller = self._get_controller(side=None)
        return self._fetch_calibration(controller, None)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def convert_mjpg(self, input_path: str, output_path: str, fps: str = "30") -> Dict[str, Any]:
        target = getattr(self._controller, "convert_mjpg", None)
        if not callable(target) and self.is_multi:
            for controller in self._multi().controllers.values():
                candidate = getattr(controller, "convert_mjpg", None)
                if callable(candidate):
                    target = candidate
                    break
        if not callable(target):
            raise TerminalCommandError("convert_mjpg is not available on this controller.")
        try:
            returncode, stderr = target(input_path, output_path, fps)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - subprocess errors
            raise TerminalCommandError(f"ffmpeg conversion failed: {exc}") from exc
        return {"returncode": returncode, "stderr": stderr}

    # ------------------------------------------------------------------ #
    # High level convenience
    # ------------------------------------------------------------------ #
    def run(self, command_line: str, **kwargs: Any) -> Any:
        """Parse a terminal command line and execute it programmatically."""
        parsed = parse_command(command_line)
        if parsed.is_error():
            raise TerminalCommandError(parsed.message or "Invalid command")

        params = parsed.params if isinstance(parsed.params, dict) else {}
        side = params.get("side")

        if parsed.action == "noop":
            return None
        if parsed.action == "help":
            return self.help()
        if parsed.action == "device_command" and parsed.device_command:
            return self.execute_device_command(parsed.device_command, side=side)
        if parsed.action == "download_file":
            return self.download_file(
                params["path"],
                side=side,
                local_filename=params.get("local_filename"),
                progress_callback=kwargs.get("progress_callback"),
                return_bytes=kwargs.get("return_bytes", True),
            )
        if parsed.action == "download_video":
            return self.download_video(
                video_dir=params["video_dir"],
                side=side,
                output=params.get("output"),
                progress_callback=kwargs.get("progress_callback"),
            )
        if parsed.action == "connect_wifi_prompt":
            password: Optional[str] = kwargs.get("password")
            password_provider: Optional[Callable[[str], str]] = kwargs.get("password_provider")
            ssid = params.get("ssid", "")
            if password is None:
                if password_provider is None:
                    raise TerminalCommandError("Password required for connectWifi command.")
                password = password_provider(ssid)
            return self.connect_wifi(ssid, password, side=side)
        if parsed.action == "convert_mjpg":
            result = self.convert_mjpg(
                params["input_path"],
                params["output_path"],
                params["fps"],
            )
            return result
        if parsed.action == "calibrate":
            return self.calibrate(
                side=side,
                samples=params.get("samples", 0),
                force=kwargs.get("force", False),
                start_with_preview=kwargs.get("start_with_preview", True),
                confirm_callback=kwargs.get("confirm_callback"),
            )
        if parsed.action == "calibration_info":
            return self.calibration_info(side=side)
        if parsed.action == "stream":
            return self.stream(
                port=params["port"],
                side=side,
                remote_host=params.get("remote_host"),
                remote_port=params.get("port"),
                quality=params.get("quality"),
                hand_tracking=params.get("hand", False),
            )
        if parsed.action == "stream_stop":
            return self.stream_stop(side=side)

        raise TerminalCommandError(f"Unhandled command action: {parsed.action}")