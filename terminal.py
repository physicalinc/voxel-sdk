#!/usr/bin/env python3
# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""Voxel Device Terminal - thin CLI wrapper around the Voxel SDK."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SDK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SDK_ROOT not in sys.path:
    sys.path.append(SDK_ROOT)

from voxel_sdk.commands import ParsedCommand, generate_help_text, parse_command
from voxel_sdk.device_controller import DeviceController


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
        response = controller.execute_device_command(parsed.device_command)
        
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
        path = parsed.params["path"]
        local_filename = parsed.params.get("local_filename") or os.path.basename(path) or "downloaded_file"
        progress_callback = _simple_progress_printer()
        print(f"Downloading {path} -> {local_filename}")
        try:
            data = controller.download_file(path, progress_callback=progress_callback)
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
        progress_callback = _simple_progress_printer()
        try:
            summary = controller.download_video(
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

    if parsed.action == "stream":
        remote_host = parsed.params.get("remote_host")
        port = parsed.params["port"]
        print("Starting local stream viewer. Close the window or press 'q' to stop.")
        try:
            controller.stream_with_visualization(
                port=port,
                remote_host=remote_host,
                remote_port=port,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Stream failed: {exc}")
        return

    if parsed.action == "stream_stop":
        try:
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
    controller: Optional[DeviceController] = None
    prompt_label = "voxel"
    
    try:
        if use_ble:
            from voxel_sdk.ble import BleVoxelTransport

            ble_name = args.ble_name or "voxel"
            transport = BleVoxelTransport(device_name=ble_name)
            target_address = args.ble_address or ""
            if target_address:
                print(f"Connecting via Bluetooth (address {target_address}, name prefix '{ble_name}')...")
            else:
                print(f"Connecting via Bluetooth (scanning for '{ble_name}')...")
            transport.connect(target_address)
        else:
            from voxel_sdk.serial import SerialVoxelTransport

            transport = SerialVoxelTransport(
                args.port, baudrate=args.baudrate, timeout=30
            )
            print(f"Connecting via wired connection on {args.port}...")
            transport.connect()

        controller = DeviceController(transport)
        try:
            info = controller.execute_device_command("get_device_name")
            if isinstance(info, dict):
                device_name = info.get("device_name")
                if isinstance(device_name, str) and device_name:
                    prompt_label = device_name.strip() or "voxel"
                    print(f"Connected to device '{device_name}'.")
        except Exception:  # noqa: BLE001
            prompt_label = "voxel"
        prompt_label = prompt_label.replace(" ", "-")
        print("Connected. Type commands and press Enter. Type 'exit' to quit.")
        print(
            "Examples: ls /, cat /test.txt, connect-wifi MySSID MyPassword, stream 203.0.113.10 9000"
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