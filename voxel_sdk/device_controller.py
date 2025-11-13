# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""High-level controller that exposes reusable Voxel device operations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple

from .voxel import VoxelFileSystem, VoxelTransport
from .commands import split_device_command

ProgressCallback = Optional[Callable[[int, str], None]]


@dataclass
class DownloadSummary:
    output_path: str
    size_bytes: int
    frames: int
    fps: int


class DeviceController:
    """Convenience wrapper for issuing commands to a Voxel device."""

    def __init__(self, transport: VoxelTransport):
        self.transport = transport
        self.filesystem = VoxelFileSystem(transport)

    def connect(self, address: str = "") -> None:
        """Connect the transport."""
        self.transport.connect(address)

    def disconnect(self) -> None:
        """Disconnect the transport."""
        self.transport.disconnect()

    def is_connected(self) -> bool:
        return self.transport.is_connected()

    def ensure_connected(self) -> None:
        if not self.is_connected():
            raise ConnectionError("Not connected to Voxel device")

    def execute_device_command(self, device_command: str) -> Dict[str, Any]:
        """Send a raw device command (already serialized)."""
        self.ensure_connected()
        command, data = split_device_command(device_command)
        return self.transport.send_command(command, data)

    def download_file(self, path: str, progress_callback: ProgressCallback = None) -> bytes:
        """Download file contents as bytes."""
        self.ensure_connected()
        return self.filesystem.download_file(path, progress_callback)

    def download_file_to_path(
        self,
        device_path: str,
        local_path: str,
        progress_callback: ProgressCallback = None,
    ) -> str:
        """Download a file and persist to disk. Returns absolute path."""
        data = self.download_file(device_path, progress_callback)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        with open(local_path, "wb") as handle:
            handle.write(data)
        return os.path.abspath(local_path)

    def stop_stream(self) -> Dict[str, Any]:
        """Stop remote streaming if active."""
        return self.execute_device_command("rdmp_stop")

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
        """Proxy to the Voxel SDK stream visualization helper."""
        self.ensure_connected()
        target_port = remote_port if remote_port is not None else port
        self.filesystem.stream_with_visualization(
            port=port,
            host=host,
            remote_host=remote_host,
            remote_port=target_port,
            hand_tracking=hand_tracking,
            quality=quality,
            window_name=window_name,
            connect_timeout=connect_timeout,
        )

    def download_video(
        self,
        video_dir: str,
        output: Optional[str] = None,
        cleanup_frames: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> DownloadSummary:
        """Download all frames for a recorded video and convert to MP4."""

        self.ensure_connected()

        metadata_path = os.path.join(video_dir, "metadata.json")
        if progress_callback:
            progress_callback(5, f"Fetching metadata from {metadata_path}")

        metadata_bytes = self.download_file(metadata_path, progress_callback=None)
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        video_name = metadata.get("video_name", "video")
        target_fps = int(metadata.get("target_fps", 30))
        frames = metadata.get("frames", [])

        if not frames:
            raise ValueError("No frames listed in metadata")

        local_dir = os.path.abspath(video_name)
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)

        for index, frame_info in enumerate(frames):
            frame_filename = frame_info["filename"]
            frame_path = os.path.join(video_dir, frame_filename)
            ffmpeg_name = f"frame_{index:010d}.jpg"
            local_frame_path = os.path.join(local_dir, ffmpeg_name)

            if progress_callback:
                percent = int((index + 1) * 70 / max(1, len(frames))) + 10
                percent = min(percent, 80)
                progress_callback(percent, f"Downloading frame {index + 1}/{len(frames)}")

            frame_bytes = self.download_file(frame_path, progress_callback=None)
            with open(local_frame_path, "wb") as handle:
                handle.write(frame_bytes)

        output_filename = output or f"{video_name}.mp4"
        output_path = os.path.abspath(output_filename)

        if progress_callback:
            progress_callback(85, "Invoking ffmpeg to create MP4...")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-r",
            str(target_fps),
            "-i",
            os.path.join(local_dir, "frame_%010d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError("ffmpeg not found. Install ffmpeg to convert MJPG to MP4") from exc

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

        size_bytes = os.path.getsize(output_path)

        if cleanup_frames:
            shutil.rmtree(local_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(100, "Video download complete")

        return DownloadSummary(
            output_path=output_path,
            size_bytes=size_bytes,
            frames=len(frames),
            fps=target_fps,
        )

    def convert_mjpg(
        self,
        input_path: str,
        output_path: str,
        fps: str = "30",
    ) -> Tuple[int, str]:
        """Convert an MJPG file to MP4 using ffmpeg. Returns (returncode, stderr)."""

        cmd = [
            "ffmpeg",
            "-y",
            "-r",
            str(fps),
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError("ffmpeg not found. Install ffmpeg to use convert_mjpg") from exc

        return result.returncode, result.stderr.strip()

