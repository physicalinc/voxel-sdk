#!/usr/bin/env python3
# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""Voxel Device Terminal - thin CLI wrapper around the Voxel SDK."""

import argparse
import asyncio
import json
import os
import sys
import time
import getpass
from typing import Any, Dict, Optional, Tuple, List

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SDK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SDK_ROOT not in sys.path:
    sys.path.append(SDK_ROOT)

from voxel_sdk.commands import ParsedCommand, generate_help_text, parse_command
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.multi_device_controller import MultiDeviceController
from voxel_sdk.calibration import calibrate_and_save, CalibrationResult

try:
    from voxel_sdk.ble import BleVoxelTransport, BleakScanner
except Exception:  # pragma: no cover
    BleVoxelTransport = None  # type: ignore
    BleakScanner = None  # type: ignore


def _run_asyncio_discover(timeout: float = 5.0):
    if BleakScanner is None:
        return []

    async def _discover():
        return await BleakScanner.discover(timeout=timeout)

    try:
        return asyncio.run(_discover())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(BleakScanner.discover(timeout=timeout))
        finally:
            loop.close()
    except Exception:
        return []


def _select_dual_devices(
    devices: List[Any],
    base_prefix: str,
    left_hint: Optional[str],
    right_hint: Optional[str],
) -> Tuple[Optional[Any], Optional[Any]]:
    def _normalize(value: Optional[str]) -> Optional[str]:
        return value.lower().strip() if value else None

    base_lower = base_prefix.lower()
    matches = []
    for device in devices:
        name = getattr(device, "name", "") or ""
        if name and name.lower().startswith(base_lower):
            matches.append(device)

    if not matches:
        return None, None

    def _find_device(
        candidates: List[Any],
        keywords: List[str],
        exclude_address: Optional[str] = None,
    ) -> Optional[Any]:
        for keyword in keywords:
            if not keyword:
                continue
            keyword = keyword.lower()
            for device in candidates:
                name = (getattr(device, "name", "") or "").lower()
                address = getattr(device, "address", None)
                if exclude_address and address == exclude_address:
                    continue
                if keyword in name:
                    return device
        return None

    left_keywords = []
    right_keywords = []

    left_keywords.append(_normalize(left_hint) or "")
    right_keywords.append(_normalize(right_hint) or "")
    left_keywords.extend(
        [
            f"{base_lower}-left",
            f"{base_lower} left",
            "left",
            "l",
        ]
    )
    right_keywords.extend(
        [
            f"{base_lower}-right",
            f"{base_lower} right",
            "right",
            "r",
        ]
    )

    left_device = _find_device(matches, left_keywords)
    right_device = _find_device(matches, right_keywords, exclude_address=getattr(left_device, "address", None))

    return left_device, right_device


def _print_directory_listing(response: Dict[str, Any]) -> None:
    files = response.get("files", [])
    if not files:
        print("No files found.")
        return

    has_timestamps = any("date_modified" in entry for entry in files)
    if has_timestamps:
        print(f"{'Name':<35} {'Type':<10} {'Size':<12} {'Modified'}")
        print("-" * 75)
        for entry in files:
            name = entry.get("name", "")
            if len(name) > 35:
                name = name[:33] + ".."
            size_text = (
                f"{entry.get('size', 0)} bytes" if entry.get("type") == "file" else "<DIR>"
            )
            modified = entry.get("date_modified", "Unknown")
            print(f"{name:<35} {entry.get('type', ''):<10} {size_text:<12} {modified}")
    else:
        print(f"{'Name':<35} {'Type':<10} {'Size'}")
        print("-" * 55)
        for entry in files:
            name = entry.get("name", "")
            if len(name) > 35:
                name = name[:33] + ".."
            size_text = (
                f"{entry.get('size', 0)} bytes" if entry.get("type") == "file" else "<DIR>"
            )
            print(f"{name:<35} {entry.get('type', ''):<10} {size_text}")


def _simple_progress_printer() -> callable:
    last_line = ""

    def callback(percent: int, message: str) -> None:
        nonlocal last_line
        line = f"{percent:3d}% {message}"
        if line != last_line:
            print(f"\r{line.ljust(80)}", end="", flush=True)
            last_line = line
        if percent >= 100:
            print()

    return callback


def _format_wifi_scan(response: Dict[str, Any]) -> None:
    """Format WiFi scan results in a readable table."""
    # Print JSON response first
    print(json.dumps(response, indent=2))
    print()
    
    if "error" in response:
        print(f"❌ WiFi Scan Failed: {response['error']}")
        return
    
    networks = response.get("networks", [])
    count = response.get("count", len(networks))
    
    print(f"\n📡 Found {count} WiFi Network(s):\n")
    print(f"{'SSID':<30} {'RSSI':<8} {'Channel':<8} {'Encryption':<15} {'Signal'}")
    print("-" * 85)
    
    # Sort by RSSI (strongest first)
    sorted_networks = sorted(networks, key=lambda x: x.get("rssi", -100), reverse=True)
    
    for network in sorted_networks:
        ssid = network.get("ssid", "Unknown")
        rssi = network.get("rssi", -100)
        channel = network.get("channel", 0)
        encryption = network.get("encryption", "UNKNOWN")
        is_open = network.get("is_open", False)
        
        # Truncate long SSIDs
        if len(ssid) > 28:
            ssid = ssid[:25] + "..."
        
        # Signal quality indicator
        if rssi > -50:
            signal = "█████ Excellent"
        elif rssi > -70:
            signal = "████  Good"
        elif rssi > -85:
            signal = "███   Fair"
        else:
            signal = "██    Poor"
        
        # Highlight open networks
        if is_open:
            encryption = f"{encryption} (OPEN)"
        
        print(f"{ssid:<30} {rssi:<8} {channel:<8} {encryption:<15} {signal}")
    
    print()


def _format_wifi_response(response: Dict[str, Any]) -> None:
    """Format WiFi connection response with detailed diagnostics."""
    # Print JSON response first
    print(json.dumps(response, indent=2))
    print()
    
    # Then print formatted output
    if "error" in response:
        print("❌ WiFi Connection Failed")
        print(f"   Error: {response['error']}")
        
        if "error_code" in response:
            print(f"   Error Code: {response['error_code']}")
        
        if "error_detail" in response:
            print(f"   Details: {response['error_detail']}")
        
        if "status" in response:
            print(f"   WiFi Status: {response['status']}")
        
        if "status_code" in response:
            print(f"   Status Code: {response['status_code']}")
        
        if "ssid" in response:
            print(f"   SSID: {response['ssid']}")
        
        if "network_found_in_scan" in response:
            found = response["network_found_in_scan"]
            print(f"   Network Found in Scan: {found}")
            
            if found and "rssi_at_scan" in response:
                print(f"   Signal Strength at Scan: {response['rssi_at_scan']} dBm")
            
            if found and "was_open" in response:
                print(f"   Network Type: {'Open' if response['was_open'] else 'Encrypted'}")
        
        if "attempts" in response:
            print(f"   Connection Attempts: {response['attempts']}")
        
        if "timeout_seconds" in response:
            print(f"   Timeout: {response['timeout_seconds']} seconds")
        
        print("\n💡 Troubleshooting Tips:")
        error_code = response.get("error_code", "")
        
        if error_code == "WRONG_PASSWORD":
            print("   • Double-check the password")
            print("   • Ensure there are no extra spaces")
            print("   • Try re-entering the password")
        elif error_code == "NO_SSID_AVAIL":
            print("   • Verify the network is 2.4GHz (device doesn't support 5GHz)")
            print("   • Check if the network is within range")
            print("   • Try moving closer to the router")
            print("   • Verify the SSID spelling is correct")
        elif error_code == "CONNECT_FAILED":
            print("   • Network may require captive portal authentication")
            print("   • Network may require enterprise/WPA2-Enterprise (not supported)")
            print("   • Try connecting from another device to verify network is working")
            print("   • Check router settings for MAC filtering or other restrictions")
        else:
            print("   • Check antenna connection")
            print("   • Verify network is 2.4GHz")
            print("   • Try a different network")
            print("   • Check router logs for connection attempts")
    else:
        print("✅ WiFi Connected Successfully")
        if "ssid" in response:
            print(f"   SSID: {response['ssid']}")
        if "ip" in response:
            print(f"   IP Address: {response['ip']}")
        if "gateway" in response:
            print(f"   Gateway: {response['gateway']}")
        if "subnet" in response:
            print(f"   Subnet: {response['subnet']}")
        if "rssi" in response:
            rssi = response["rssi"]
            signal_quality = "Excellent" if rssi > -50 else "Good" if rssi > -70 else "Fair" if rssi > -85 else "Poor"
            print(f"   Signal Strength: {rssi} dBm ({signal_quality})")
        if "mac" in response:
            print(f"   MAC Address: {response['mac']}")


def _handle_parsed_command(
    controller: DeviceController,
    parsed: ParsedCommand,
) -> None:
    if parsed.action == "help":
        print(generate_help_text())
        return

    if parsed.action == "device_command" and parsed.device_command:
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        is_multi = getattr(controller, "is_multi", False)
        if is_multi and side in ("left", "right"):
            # Route to a single device
            try:
                sub = controller.get_controller(side)  # type: ignore[attr-defined]
                if sub is None:
                    print(f"No '{side}' device is connected.")
                    return
                response = sub.execute_device_command(parsed.device_command)
            except Exception as exc:  # noqa: BLE001
                print(f"Command failed on {side}: {exc}")
                return
            # Pretty print single-device responses
            if parsed.device_command.startswith("connectWifi:"):
                if isinstance(response, dict):
                    _format_wifi_response(response)
                else:
                    print(json.dumps(response, indent=2))
            elif parsed.device_command == "scanWifi" and isinstance(response, dict):
                _format_wifi_scan(response)
            elif parsed.device_command.startswith("list_dir:") and isinstance(response, dict):
                _print_directory_listing(response)
            else:
                print(json.dumps(response, indent=2))
        else:
            response = controller.execute_device_command(parsed.device_command)
            if getattr(controller, "is_multi", False):
                print(json.dumps(response, indent=2))
            else:
                # Special formatting for WiFi commands
                if parsed.device_command.startswith("connectWifi:"):
                    if isinstance(response, dict):
                        _format_wifi_response(response)
                    else:
                        print(json.dumps(response, indent=2))
                elif parsed.device_command == "scanWifi" and isinstance(response, dict):
                    _format_wifi_scan(response)
                elif parsed.device_command.startswith("list_dir:") and isinstance(response, dict):
                    _print_directory_listing(response)
                else:
                    print(json.dumps(response, indent=2))
        return

    if parsed.action == "download_file":
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        if getattr(controller, "is_multi", False) and side not in ("left", "right"):
            print("⚠️  File downloads in dual mode require selecting a side: add --left or --right.")
            return
        # Select appropriate controller
        if getattr(controller, "is_multi", False) and side in ("left", "right"):
            sub = controller.get_controller(side)  # type: ignore[attr-defined]
            if sub is None:
                print(f"No '{side}' device is connected.")
                return
            active = sub
        else:
            active = controller
        path = parsed.params["path"]
        local_filename = parsed.params.get("local_filename") or os.path.basename(path) or "downloaded_file"
        progress_callback = _simple_progress_printer()
        print(f"Downloading {path} -> {local_filename}")
        try:
            data = active.download_file(path, progress_callback=progress_callback)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}")
            return

        if not local_filename:
            local_filename = f"downloaded_{int(time.time())}"

        local_path = os.path.abspath(local_filename)
        with open(local_path, "wb") as handle:
            handle.write(data)

        size_kb = len(data) / 1024
        size_text = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
        print(f"Saved {len(data)} bytes ({size_text}) to {local_path}")
        return

    if parsed.action == "download_video":
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        if getattr(controller, "is_multi", False) and side not in ("left", "right"):
            print("⚠️  Video downloads in dual mode require selecting a side: add --left or --right.")
            return
        if getattr(controller, "is_multi", False) and side in ("left", "right"):
            sub = controller.get_controller(side)  # type: ignore[attr-defined]
            if sub is None:
                print(f"No '{side}' device is connected.")
                return
            active = sub
        else:
            active = controller
        progress_callback = _simple_progress_printer()
        try:
            summary = active.download_video(  # type: ignore[attr-defined]
                video_dir=parsed.params["video_dir"],
                output=parsed.params.get("output"),
                progress_callback=progress_callback,
            )
            size_mb = summary.size_bytes / (1024 * 1024)
            print(f"Video saved to {summary.output_path} ({size_mb:.1f} MB, {summary.frames} frames @ {summary.fps} FPS)")
        except FileNotFoundError as exc:
            print(f"ffmpeg not found: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"Video download failed: {exc}")
        return
    
    if parsed.action == "connect_wifi_prompt":
        ssid = parsed.params.get("ssid", "") if isinstance(parsed.params, dict) else ""
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        if not ssid:
            print("SSID is required. Usage: connect-wifi <ssid> [password]")
            return
        try:
            print(f"SSID: {ssid}")
            password = getpass.getpass("Password (leave blank for open network): ")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to read password: {exc}")
            return
        try:
            device_command = f"connectWifi:{ssid}|{password}"
            is_multi = getattr(controller, "is_multi", False)
            if is_multi and side in ("left", "right"):
                sub = controller.get_controller(side)  # type: ignore[attr-defined]
                if sub is None:
                    print(f"No '{side}' device is connected.")
                    return
                response = sub.execute_device_command(device_command)
                if isinstance(response, dict):
                    _format_wifi_response(response)
                else:
                    print(json.dumps(response, indent=2))
            else:
                response = controller.execute_device_command(device_command)
                if getattr(controller, "is_multi", False):
                    print(json.dumps(response, indent=2))
                else:
                    if isinstance(response, dict):
                        _format_wifi_response(response)
                    else:
                        print(json.dumps(response, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"WiFi connection failed: {exc}")
        return

    if parsed.action == "convert_mjpg":
        try:
            returncode, stderr = controller.convert_mjpg(
                parsed.params["input_path"],
                parsed.params["output_path"],
                parsed.params["fps"],
            )
        except FileNotFoundError as exc:
            print(f"ffmpeg not found: {exc}")
            return

        if returncode == 0:
            print(f"Converted {parsed.params['input_path']} -> {parsed.params['output_path']}")
        else:
            print(f"Conversion failed (code {returncode}): {stderr}")
        return
    
    if parsed.action == "calibrate":
        # Determine requested sample count, if provided
        samples = int(parsed.params.get("samples", 0)) if isinstance(parsed.params, dict) else 0
        target_side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        is_multi = getattr(controller, "is_multi", False)
        try:
            if is_multi:
                # Calibrate sequentially for each device (left then right if present)
                multi: MultiDeviceController = controller  # type: ignore[assignment]
                labels = list(multi.device_labels.keys())
                # Prefer left/right order if available
                ordered = []
                for key in ("left", "right"):
                    if key in labels:
                        ordered.append(key)
                for lb in labels:
                    if lb not in ordered:
                        ordered.append(lb)
                # Filter to specific side if requested
                if target_side in ("left", "right"):
                    ordered = [lb for lb in ordered if lb == target_side]
                    if not ordered:
                        print(f"No '{target_side}' device is connected.")
                        return
                results: Dict[str, Dict[str, Any]] = {}
                for lb in ordered:
                    name = multi.device_labels.get(lb, lb)
                    sub = multi.get_controller(lb)
                    if sub is None:
                        continue
                    print(f"\n=== Calibrating {lb} ({name}) ===")
                    print("A full-screen calibration pattern will appear.")
                    print("Point the camera at the screen from different angles and distances.")
                    print("It auto-captures; keys: 'p' toggle PiP preview, 'c' capture, 'i' invert grid, 'q' finish.")
                    print("A small live preview will appear in the bottom-right corner (won't cover the grid).")
                    result: CalibrationResult = calibrate_and_save(sub, label=lb, requested_samples=samples, start_with_preview=True)
                    results[lb] = {
                        "saved_to": result.device_path,
                        "rms": result.rms,
                        "image_size": {"w": result.image_size[0], "h": result.image_size[1]},
                        "timestamp": result.timestamp,
                    }
                    print(f"✓ Saved {lb} intrinsics to {result.device_path} (RMS error {result.rms:.4f})")
                if results:
                    print("\nCalibration complete:")
                    print(json.dumps(results, indent=2))
            else:
                print("=== Calibrating device ===")
                print("A full-screen calibration pattern will appear.")
                print("Point the camera at the screen from different angles and distances.")
                print("It auto-captures; keys: 'p' toggle PiP preview, 'c' capture, 'i' invert grid, 'q' finish.")
                print("A small live preview will appear in the bottom-right corner (won't cover the grid).")
                result: CalibrationResult = calibrate_and_save(controller, requested_samples=samples, start_with_preview=True)
                summary = {
                    "saved_to": result.device_path,
                    "rms": result.rms,
                    "image_size": {"w": result.image_size[0], "h": result.image_size[1]},
                    "timestamp": result.timestamp,
                }
                print(f"✓ Saved intrinsics to {result.device_path} (RMS error {result.rms:.4f})")
                print(json.dumps(summary, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"Calibration failed: {exc}")
        return

    if parsed.action == "stream":
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        remote_host = parsed.params.get("remote_host")
        port = parsed.params["port"]
        quality = parsed.params.get("quality")
        hand_tracking = parsed.params.get("hand", False)
        is_multi = getattr(controller, "is_multi", False)
        if is_multi and side not in ("left", "right"):
            label_map = getattr(controller, "device_labels", {})
            names = ", ".join(f"{label}:{name}" for label, name in label_map.items()) or "devices"
            print(
                f"Starting dual stream viewers for {names}. "
                f"Ports {port}, {port + 1} will be used. Close each window or press 'q' to stop."
            )
            if quality is not None:
                print(f"Requested JPEG quality: {quality} (0=highest fidelity, 63=lowest).")
            if hand_tracking:
                print("Hand tracking overlay enabled (MediaPipe).")
        else:
            if is_multi and side in ("left", "right"):
                sub = controller.get_controller(side)  # type: ignore[attr-defined]
                if sub is None:
                    print(f"No '{side}' device is connected.")
                    return
                controller = sub  # type: ignore[assignment]
            print("Starting local stream viewer. Close the window or press 'q' to stop.")
            if quality is not None:
                print(f"Requested device JPEG quality: {quality} (0=highest fidelity, 63=lowest).")
            if hand_tracking:
                print("Hand tracking overlay enabled (MediaPipe).")
        try:
            controller.stream_with_visualization(
                port=port,
                remote_host=remote_host,
                remote_port=port,
                quality=quality,
                hand_tracking=hand_tracking,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Stream failed: {exc}")
        return

    if parsed.action == "stream_stop":
        side = parsed.params.get("side") if isinstance(parsed.params, dict) else None
        is_multi = getattr(controller, "is_multi", False)
        try:
            if is_multi and side in ("left", "right"):
                sub = controller.get_controller(side)  # type: ignore[attr-defined]
                if sub is None:
                    print(f"No '{side}' device is connected.")
                    return
                response = sub.stop_stream()
                print(json.dumps(response, indent=2))
            else:
                response = controller.stop_stream()
                print(json.dumps(response, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to stop stream: {exc}")
        return

    print(parsed.message or "Nothing to do")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxel Device Terminal")
    parser.add_argument(
        "--port",
        "-p",
        default="/dev/cu.usbmodem1101",
        help="Wired serial port (default: /dev/cu.usbmodem1101)",
    )
    parser.add_argument(
        "--baudrate",
        "-b",
        type=int,
        default=115200,
        help="Baudrate for wired transport (default: 115200)",
    )
    parser.add_argument(
        "--transport",
        choices=["prompt", "serial", "ble"],
        default="prompt",
        help="Select transport mode (default: prompt for choice)",
    )
    parser.add_argument(
        "--ble-name",
        default=None,
        help="Bluetooth device name prefix to match (default: voxel)",
    )
    parser.add_argument(
        "--ble-address",
        default="",
        help="Optional Bluetooth MAC address to connect directly",
    )
    parser.add_argument(
        "--ble-left-name",
        default=None,
        help="Optional Bluetooth name prefix/keyword for the left unit (defaults to detecting '*left').",
    )
    parser.add_argument(
        "--ble-right-name",
        default=None,
        help="Optional Bluetooth name prefix/keyword for the right unit (defaults to detecting '*right').",
    )
    parser.add_argument(
        "--disable-dual",
        action="store_true",
        help="Force single-device mode even if left/right Bluetooth devices are detected.",
    )
    
    args = parser.parse_args()
    
    transport_choice = args.transport
    if transport_choice == "prompt":
        print("Select connection method:")
        print("  1) Wired")
        print("  2) Bluetooth")
        choice = input("Enter 1 or 2 [2]: ").strip() or "2"
        use_ble = choice == "2"
    else:
        use_ble = transport_choice == "ble"
    
    transport = None
    controller: Optional[Any] = None
    prompt_label = "voxel"
    
    try:
        if use_ble:
            ble_name = args.ble_name or "voxel"
            if BleVoxelTransport is None:
                raise RuntimeError("BLE support requires the 'bleak' package. Install voxel-sdk[ble] to enable it.")

            dual_ctrl: Optional[MultiDeviceController] = None
            left_transport = None
            right_transport = None

            if (
                not args.disable_dual
                and not args.ble_address
                and BleakScanner is not None
            ):
                devices = _run_asyncio_discover()
                left_device, right_device = _select_dual_devices(
                    devices,
                    ble_name,
                    args.ble_left_name,
                    args.ble_right_name,
                )
                if left_device and right_device:
                    left_name = getattr(left_device, "name", "") or f"{ble_name}-left"
                    right_name = getattr(right_device, "name", "") or f"{ble_name}-right"
                    print(f"Detected '{left_name}' and '{right_name}'. Attempting synchronized connection...")
                    left_transport = BleVoxelTransport(device_name=left_name)
                    right_transport = BleVoxelTransport(device_name=right_name)
                    try:
                        left_transport.connect(getattr(left_device, "address", ""))
                        right_transport.connect(getattr(right_device, "address", ""))
                        left_controller = DeviceController(left_transport)
                        right_controller = DeviceController(right_transport)
                        dual_ctrl = MultiDeviceController(
                            {"left": left_controller, "right": right_controller},
                            display_names={
                                "left": getattr(left_device, "name", "") or "left",
                                "right": getattr(right_device, "name", "") or "right",
                            },
                        )
                        controller = dual_ctrl
                        prompt_label = "dual"
                        print("Connected to BLE pair:")
                        for label, name in dual_ctrl.device_labels.items():
                            print(f"  {label}: {name}")
                    except Exception as exc:
                        print(f"⚠️  Failed to connect to both devices ({exc}). Falling back to single-device mode.")
                        if left_transport:
                            try:
                                left_transport.disconnect()
                            except Exception:
                                pass
                        if right_transport:
                            try:
                                right_transport.disconnect()
                            except Exception:
                                pass
                        controller = None

            if controller is None:
                transport = BleVoxelTransport(device_name=ble_name)
                target_address = args.ble_address or ""
                if target_address:
                    print(f"Connecting via Bluetooth (address {target_address}, name prefix '{ble_name}')...")
                else:
                    print(f"Connecting via Bluetooth (scanning for '{ble_name}')...")
                transport.connect(target_address)
                controller = DeviceController(transport)
        else:
            from voxel_sdk.serial import SerialVoxelTransport

            transport = SerialVoxelTransport(
                args.port, baudrate=args.baudrate, timeout=30
            )
            print(f"Connecting via wired connection on {args.port}...")
            transport.connect()
            controller = DeviceController(transport)

        if controller is None:
            raise RuntimeError("Failed to establish controller connection.")

        if getattr(controller, "is_multi", False):
            try:
                info = controller.execute_device_command("get_device_name")
                if isinstance(info, dict):
                    responses = info.get("responses", {})
                    if isinstance(responses, dict):
                        for label, payload in responses.items():
                            if isinstance(payload, dict):
                                device_name = payload.get("device_name")
                                if isinstance(device_name, str) and device_name:
                                    print(f"{label} device name: {device_name}")
            except Exception:
                pass
            prompt_label = "dual"
        else:
            try:
                info = controller.execute_device_command("get_device_name")
                if isinstance(info, dict):
                    device_name = info.get("device_name")
                    if isinstance(device_name, str) and device_name:
                        prompt_label = device_name.strip() or "voxel"
                        print(f"Connected to device '{device_name}'.")
            except Exception:  # noqa: BLE001
                prompt_label = "voxel"

        prompt_label = (prompt_label or "voxel").replace(" ", "-")
        print("Connected. Type commands and press Enter. Type 'exit' to quit.")
        print(
            "Examples: ls /, cat /test.txt, connect-wifi MySSID MyPassword, stream 203.0.113.10 9000 10 --hand"
        )
        print("-" * 60)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to connect: {exc}")
        if transport:
            try:
                transport.disconnect()
            except Exception:  # noqa: BLE001
                pass
        return
    
    assert controller is not None
    
    try:
        while True:
            try:
                raw_command = input(f"{prompt_label}> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            
            if raw_command.lower() in {"exit", "quit", "q"}:
                break
            
            if raw_command:
                parsed = parse_command(raw_command)
                if parsed.is_error():
                    print(parsed.message)
                elif parsed.action != "noop":
                    _handle_parsed_command(controller, parsed)
    finally:
        try:
            controller.disconnect()
        except Exception:  # noqa: BLE001
            pass
        print("Disconnected.")


if __name__ == "__main__":
    main()