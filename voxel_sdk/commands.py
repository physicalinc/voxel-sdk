# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""Shared command parsing utilities for Voxel device interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


COMMAND_ALIASES: Dict[str, str] = {
    "card_info": "card_info",
    "cardinfo": "card_info",
    "df": "card_info",
    "list_dir": "list_dir",
    "listdir": "list_dir",
    "ls": "list_dir",
    "dir": "list_dir",
    "read_file": "read_file",
    "readfile": "read_file",
    "cat": "read_file",
    "write_file": "write_file",
    "writefile": "write_file",
    "append_file": "append_file",
    "appendfile": "append_file",
    "append": "append_file",
    "delete_file": "delete_file",
    "deletefile": "delete_file",
    "rm": "delete_file",
    "file_exists": "file_exists",
    "exists": "file_exists",
    "file_size": "file_size",
    "size": "file_size",
    "stat": "file_size",
    "create_dir": "create_dir",
    "mkdir": "create_dir",
    "remove_dir": "remove_dir",
    "rmdir": "remove_dir",
    "rename_file": "rename_file",
    "rename": "rename_file",
    "mv": "rename_file",
    "download_file": "download_file",
    "download": "download_file",
    "get": "download_file",
    "download_video": "download_video",
    "downloadvideo": "download_video",
    "convert_mjpg": "convert_mjpg",
    "convertmjpg": "convert_mjpg",
    "ping_host": "ping_host",
    "ping": "ping_host",
    "rdmp_stream": "rdmp_stream",
    "stream": "rdmp_stream",
    "start_stream": "rdmp_stream",
    "rdmp_stop": "rdmp_stop",
    "stop_stream": "rdmp_stop",
    "stream-stop": "rdmp_stop",
    "camera_status": "camera_status",
    "camera_capture": "camera_capture",
    "camera_record": "camera_record",
    "camera_stop": "camera_stop",
    "camera_config": "camera_config",
    "camera_reset": "camera_reset",
    "connectwifi": "connectWifi",
    "connect_wifi": "connectWifi",
    "wifi_connect": "connectWifi",
    "disconnectwifi": "disconnectWifi",
    "disconnect_wifi": "disconnectWifi",
    "wifi_disconnect": "disconnectWifi",
    "scanwifi": "scanWifi",
    "scan_wifi": "scanWifi",
    "wifi_scan": "scanWifi",
    "listwifi": "scanWifi",
    "list_wifi": "scanWifi",
    "wifi_list": "scanWifi",
    "help": "help",
    "?": "help",
}

COMMAND_DISPLAY_ALIASES: Dict[str, str] = {
    "card_info": "df",
    "list_dir": "ls",
    "read_file": "cat",
    "write_file": "write-file",
    "append_file": "append",
    "delete_file": "rm",
    "file_exists": "exists",
    "file_size": "stat",
    "create_dir": "mkdir",
    "remove_dir": "rmdir",
    "rename_file": "mv",
    "download_file": "download",
    "download_video": "download-video",
    "convert_mjpg": "convert-mjpg",
    "ping_host": "ping",
    "rdmp_stream": "stream",
    "rdmp_stop": "stream-stop",
    "camera_status": "camera-status",
    "camera_capture": "camera-capture",
    "camera_record": "camera-record",
    "camera_stop": "camera-stop",
    "camera_config": "camera-config",
    "camera_reset": "camera-reset",
    "connectWifi": "connect-wifi",
    "disconnectWifi": "disconnect-wifi",
    "scanWifi": "scan-wifi",
    "help": "?",
}

HELP_COLUMN_WIDTH = 35

HELP_SECTIONS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    (
        "Filesystem",
        [
            ("card_info", "", "Get SD card information"),
            ("list_dir", "[path]", "List directory (default: /)"),
            ("read_file", "<path>", "Read file contents"),
            ("write_file", "<path> <content>", "Write content to file"),
            ("append_file", "<path> <content>", "Append content to file"),
            ("delete_file", "<path>", "Delete file"),
            ("file_exists", "<path>", "Check if file exists"),
            ("file_size", "<path>", "Get file size"),
            ("create_dir", "<path>", "Create directory"),
            ("remove_dir", "<path>", "Remove directory"),
            ("rename_file", "<old_path> <new_path>", "Rename/move file"),
            ("download_file", "<path> [local_name]", "Download file from device"),
            ("download_video", "<video_dir> [output.mp4]", "Download frames and convert to MP4"),
        ],
    ),
    (
        "Camera",
        [
            ("camera_status", "", "Get camera status"),
            ("camera_capture", "[dir] [name] [res]", "Capture photo"),
            ("camera_record", "[dir] [name] [res] [fps]", "Start video recording"),
            ("camera_stop", "", "Stop video recording"),
            ("camera_config", "[res] [quality] [format] [fb_count]", "Configure camera"),
            ("camera_reset", "", "Reset camera"),
        ],
    ),
    (
        "Connectivity",
        [
            ("scanWifi", "", "Scan for available WiFi networks"),
            ("connectWifi", "<ssid> [password]", "Connect device to WiFi"),
            ("disconnectWifi", "", "Disconnect device from WiFi"),
            ("ping_host", "<host> [count]", "Ping a host from the device"),
            ("rdmp_stream", "<host> [port] [quality] [--hand]", "Stream MJPEG frames to remote host (quality 0-63; add --hand for MediaPipe overlay)"),
            ("rdmp_stop", "", "Stop remote stream"),
        ],
    ),
    (
        "Utility",
        [
            ("convert_mjpg", "<mjpg_file> [output.mp4] [fps]", "Convert MJPG to MP4 using ffmpeg"),
            ("help", "", "Show this help"),
        ],
    ),
]


@dataclass
class ParsedCommand:
    """Normalized representation of a CLI command."""

    action: str
    device_command: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None

    def is_error(self) -> bool:
        return self.action == "error"


def normalize_command(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return COMMAND_ALIASES.get(key, name)


def command_label(name: str) -> str:
    alias = COMMAND_DISPLAY_ALIASES.get(name, "")
    return f"{name} ({alias})" if alias else name


def _ensure_leading_slash(path: str) -> str:
    if path and not path.startswith("/"):
        return f"/{path}"
    return path or "/"


def split_device_command(command: str) -> Tuple[str, Optional[str]]:
    if ":" in command:
        head, tail = command.split(":", 1)
        return head, tail
    return command, None


def parse_command(command_line: str) -> ParsedCommand:
    stripped = command_line.strip()
    if not stripped:
        return ParsedCommand(action="noop")

    parts = stripped.split()
    original_cmd = parts[0]
    cmd = normalize_command(original_cmd)

    if cmd == "help":
        return ParsedCommand(action="help")

    if cmd == "rdmp_stream":
        args = parts[1:]
        hand = False
        filtered_args = []
        for arg in args:
            if arg == "--hand":
                hand = True
            else:
                filtered_args.append(arg)
        args = filtered_args
        remote_host: Optional[str] = None
        port: Optional[int] = None
        quality: Optional[int] = None

        def _parse_int(value: str, label: str) -> Optional[int]:
            if value.strip() == "":
                return None
            if value.isdigit():
                return int(value)
            raise ValueError(f"Invalid {label}")

        try:
            if args:
                first = args[0]
                if first.isdigit():
                    port = int(first)
                    if len(args) > 1:
                        quality = _parse_int(args[1], "quality")
                    if len(args) > 2:
                        raise ValueError("Too many arguments for stream command")
                else:
                    remote_host = first
                    if len(args) > 1:
                        port = _parse_int(args[1], "port")
                    if len(args) > 2:
                        quality = _parse_int(args[2], "quality")
                    if len(args) > 3:
                        raise ValueError("Too many arguments for stream command")
            if port is None:
                port = 9000
            if quality is not None and not (0 <= quality <= 63):
                return ParsedCommand(action="error", message="Quality must be between 0 and 63")
        except ValueError as exc:
            return ParsedCommand(
                action="error",
                message=f"{exc}. Usage: {command_label('rdmp_stream')} <host> [port] [quality]",
            )

        return ParsedCommand(
            action="stream",
            params={"remote_host": remote_host, "port": port, "quality": quality, "hand": hand},
        )

    if cmd == "rdmp_stop":
        return ParsedCommand(action="stream_stop")

    if cmd == "download_video":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('download_video')} <video_directory> [output_filename]",
            )

        video_dir = _ensure_leading_slash(parts[1])
        output_file = parts[2] if len(parts) > 2 else None

        return ParsedCommand(
            action="download_video",
            params={"video_dir": video_dir, "output": output_file},
        )

    if cmd == "download_file":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('download_file')} <path> [local_filename]",
            )

        device_path = _ensure_leading_slash(parts[1])
        local_filename = parts[2] if len(parts) > 2 else None
        if local_filename == ".":
            local_filename = None

        return ParsedCommand(
            action="download_file",
            params={"path": device_path, "local_filename": local_filename},
        )

    if cmd == "convert_mjpg":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('convert_mjpg')} <mjpg_file> [output.mp4] [fps]",
            )

        mjpg_file = parts[1]
        output_file = parts[2] if len(parts) > 2 else mjpg_file.replace(".mjpg", ".mp4")
        fps = parts[3] if len(parts) > 3 else "30"

        return ParsedCommand(
            action="convert_mjpg",
            params={"input_path": mjpg_file, "output_path": output_file, "fps": fps},
        )

    if cmd == "list_dir":
        path = _ensure_leading_slash(parts[1] if len(parts) > 1 else "/")
        return ParsedCommand(action="device_command", device_command=f"list_dir:{path}")

    if cmd == "write_file":
        if len(parts) < 3:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('write_file')} <path> <content>",
            )
        path = _ensure_leading_slash(parts[1])
        content = " ".join(parts[2:])
        return ParsedCommand(action="device_command", device_command=f"write_file:{path}|{content}")

    if cmd == "append_file":
        if len(parts) < 3:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('append_file')} <path> <content>",
            )
        path = _ensure_leading_slash(parts[1])
        content = " ".join(parts[2:])
        return ParsedCommand(action="device_command", device_command=f"append_file:{path}|{content}")

    if cmd == "read_file":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('read_file')} <path>",
            )
        path = _ensure_leading_slash(parts[1])
        return ParsedCommand(action="device_command", device_command=f"read_file:{path}")

    if cmd == "delete_file":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('delete_file')} <path>",
            )
        return ParsedCommand(action="device_command", device_command=f"delete_file:{parts[1]}")

    if cmd == "file_exists":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('file_exists')} <path>",
            )
        return ParsedCommand(action="device_command", device_command=f"file_exists:{parts[1]}")

    if cmd == "file_size":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('file_size')} <path>",
            )
        return ParsedCommand(action="device_command", device_command=f"file_size:{parts[1]}")

    if cmd == "create_dir":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('create_dir')} <path>",
            )
        return ParsedCommand(action="device_command", device_command=f"create_dir:{parts[1]}")

    if cmd == "remove_dir":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('remove_dir')} <path>",
            )
        return ParsedCommand(action="device_command", device_command=f"remove_dir:{parts[1]}")

    if cmd == "rename_file":
        if len(parts) < 3:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('rename_file')} <old_path> <new_path>",
            )
        old_path = parts[1]
        new_path = parts[2]
        return ParsedCommand(action="device_command", device_command=f"rename_file:{old_path},{new_path}")

    if cmd == "card_info":
        return ParsedCommand(action="device_command", device_command="card_info")

    if cmd == "connectWifi":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('connectWifi')} <ssid> [password]",
            )
        ssid = parts[1]
        password = " ".join(parts[2:]) if len(parts) > 2 else ""
        return ParsedCommand(action="device_command", device_command=f"connectWifi:{ssid}|{password}")

    if cmd == "disconnectWifi":
        return ParsedCommand(action="device_command", device_command="disconnectWifi")

    if cmd == "scanWifi":
        return ParsedCommand(action="device_command", device_command="scanWifi")

    if cmd == "ping_host":
        if len(parts) < 2:
            return ParsedCommand(
                action="error",
                message=f"Usage: {command_label('ping_host')} <host> [count]",
            )
        host = parts[1]
        count = parts[2] if len(parts) > 2 else ""
        data = f"{host}|{count}" if count else host
        return ParsedCommand(action="device_command", device_command=f"ping_host:{data}")

    if cmd == "camera_status":
        return ParsedCommand(action="device_command", device_command="camera_status")

    if cmd == "camera_capture":
        directory = parts[1] if len(parts) > 1 else "/"
        name = parts[2] if len(parts) > 2 else ""
        resolution = parts[3] if len(parts) > 3 else "1600x1200"
        return ParsedCommand(
            action="device_command",
            device_command=f"camera_capture:{directory}|{name}|{resolution}",
        )

    if cmd == "camera_record":
        if len(parts) == 1:
            return ParsedCommand(action="device_command", device_command="camera_record:/")
        directory = parts[1] if len(parts) > 1 else "/"
        name = parts[2] if len(parts) > 2 else ""
        resolution = parts[3] if len(parts) > 3 else "800x600"
        fps = parts[4] if len(parts) > 4 else "30"
        return ParsedCommand(
            action="device_command",
            device_command=f"camera_record:{directory}|{name}|{resolution}|{fps}",
        )

    if cmd == "camera_stop":
        return ParsedCommand(action="device_command", device_command="camera_stop")

    if cmd == "camera_config":
        resolution = parts[1] if len(parts) > 1 else "1600x1200"
        quality = parts[2] if len(parts) > 2 else "12"
        fmt = parts[3] if len(parts) > 3 else "JPEG"
        fb_count = parts[4] if len(parts) > 4 else "1"
        return ParsedCommand(
            action="device_command",
            device_command=f"camera_config:{resolution}|{quality}|{fmt}|{fb_count}",
        )

    if cmd == "camera_reset":
        return ParsedCommand(action="device_command", device_command="camera_reset")

    return ParsedCommand(
        action="error",
        message=f"Unknown command: {original_cmd}. Type 'help' for available commands.",
    )


def generate_help_text() -> str:
    lines: List[str] = ["", "=== Voxel Device Commands ==="]
    for section, entries in HELP_SECTIONS:
        lines.append(f"=== {section} ===")
        for name, args, description in entries:
            label = command_label(name)
            command_text = f"{label} {args}".strip()
            padding = HELP_COLUMN_WIDTH - len(command_text)
            if padding < 1:
                padding = 1
            lines.append(f"{command_text}{' ' * padding}- {description}")
        lines.append("")

    lines.append("Camera resolutions: 1600x1200, 1280x720, 800x600, 640x480, etc.")
    lines.append("Default FPS: 30 (range: 1-60)")
    lines.append("exit, quit, q".ljust(HELP_COLUMN_WIDTH) + "- Exit terminal")
    lines.append("")
    return "\n".join(lines)

