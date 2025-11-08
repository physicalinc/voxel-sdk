"""BLE transport implementation for Voxel devices using Bleak."""

import asyncio
import base64
import json
import threading
from typing import Dict, Any, Optional

from .voxel import VoxelTransport

try:
    from bleak import BleakClient, BleakScanner
except Exception as e:  # pragma: no cover
    BleakClient = None  # type: ignore
    BleakScanner = None  # type: ignore


class BleVoxelTransport(VoxelTransport):
    """BLE communication transport for Voxel devices (Nordic UART-like service).

    Notes:
    - Text service UUIDs mirror firmware defaults (UART style):
      SERVICE_UUID, CHARACTERISTIC_UUID_RX (write), CHARACTERISTIC_UUID_TX (notify)
    - Binary service UUIDs: BINARY_SERVICE_UUID, BINARY_RX_UUID (write), BINARY_TX_UUID (notify)
    - All operations are provided via synchronous wrappers over Bleak's asyncio APIs.
    """

    SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    CHARACTERISTIC_UUID_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

    BINARY_SERVICE_UUID = "12345678-1234-1234-1234-123456789ABC"
    BINARY_TX_UUID = "12345678-1234-1234-1234-123456789ABD"
    BINARY_RX_UUID = "12345678-1234-1234-1234-123456789ABE"

    def __init__(self, device_name: str = "voxel"):
        self.device_name = device_name
        self.client: Optional[BleakClient] = None
        self._notify_buffer = bytearray()
        self._binary_buffer = bytearray()
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_running = False

    # ---------- Public API (sync wrappers) ----------
    def connect(self, address: str) -> None:
        # Start persistent loop first if needed
        self._start_loop_thread()
        self._run(self._connect_async(address))

    def disconnect(self) -> None:
        self._run(self._disconnect_async())

    def is_connected(self) -> bool:
        return bool(self.client) and self._connected

    def send_command(self, command: str, data: Optional[str] = None) -> Dict[str, Any]:
        cmd = command if data is None else f"{command}:{data}"
        cmd += "\n"
        return self._run(self._send_command_async(cmd))

    def send_binary_data(self, data: bytes) -> Dict[str, Any]:
        return self._run(self._send_binary_async(data))

    def receive_binary_data(self, max_bytes: int = 1024) -> bytes:
        return self._run(self._receive_binary_async(max_bytes))

    def download_file(self, path: str, progress_callback=None) -> bytes:
        return self._run(self._download_file_async(path, progress_callback))

    # ---------- Async implementation ----------
    async def _connect_async(self, address: str) -> None:
        if BleakClient is None or BleakScanner is None:
            raise RuntimeError("bleak is required for BLE support. Install with 'pip install bleak'.")

        target_address = address
        if not target_address:
            print(f"Scanning for Bluetooth devices (looking for '{self.device_name}')...")
            devices = await BleakScanner.discover()
            
            # Try to find matching device (case-insensitive)
            device_name_lower = self.device_name.lower()
            voxel = next((d for d in devices if (d.name or "").lower().startswith(device_name_lower)), None)
            if not voxel:
                error_msg = f"No Bluetooth device named '{self.device_name}' found"
                if devices:
                    error_msg += f"\nScanned {len(devices)} device(s) but none matched '{self.device_name}'"
                    # Only show device names in error message, not full list
                    available_names = [d.name for d in devices if d.name]
                    if available_names:
                        error_msg += "\nAvailable device names: " + ", ".join(available_names[:10])  # Limit to first 10
                        if len(available_names) > 10:
                            error_msg += f" ... and {len(available_names) - 10} more"
                raise ConnectionError(error_msg)
            target_address = voxel.address

        self.client = BleakClient(target_address)
        await self.client.connect()
        # Support Bleak versions where is_connected is a property vs. coroutine
        try:
            ic = getattr(self.client, "is_connected")
            if callable(ic):
                self._connected = await ic()
            else:
                self._connected = bool(ic)
        except Exception:
            # Fallback: assume connected if no exception from connect()
            self._connected = True
        if not self._connected:
            raise ConnectionError(f"Failed to connect to Bluetooth device at {target_address}")

        # Subscribe to text TX characteristic
        await self.client.start_notify(self.CHARACTERISTIC_UUID_TX, self._on_text_notify)

        # Subscribe to binary TX if available
        try:
            await self.client.start_notify(self.BINARY_TX_UUID, self._on_binary_notify)
        except Exception:
            # Binary service may not exist yet; ignore
            pass

    async def _disconnect_async(self) -> None:
        if self.client:
            try:
                try:
                    await self.client.stop_notify(self.CHARACTERISTIC_UUID_TX)
                except Exception:
                    pass
                try:
                    await self.client.stop_notify(self.BINARY_TX_UUID)
                except Exception:
                    pass
                await self.client.disconnect()
            finally:
                self.client = None
                self._connected = False
                self._notify_buffer.clear()
                self._binary_buffer.clear()
                self._stop_loop_thread()

    async def _send_command_async(self, cmd_with_newline: str) -> Dict[str, Any]:
        self._ensure_connected()
        assert self.client

        # Clear any stale data from previous commands before sending new command
        # Give a small moment for any pending notifications to arrive, then clear
        await asyncio.sleep(0.05)
        self._notify_buffer.clear()
        
        # Send the command
        await self.client.write_gatt_char(self.CHARACTERISTIC_UUID_RX, cmd_with_newline.encode("utf-8"), response=False)
        
        # Small delay to ensure command is processed
        await asyncio.sleep(0.05)
        
        # Wait for a newline-terminated response with a timeout that adapts to the command
        command_line = cmd_with_newline.strip()
        timeout_seconds = 5.0
        if command_line:
            separator_index = command_line.find(":")
            command_name = command_line if separator_index == -1 else command_line[:separator_index]
            command_args = "" if separator_index == -1 else command_line[separator_index + 1:]

            if command_name == "connectWifi":
                # Association/auth and DHCP can routinely take 15-20 seconds
                # Firmware timeout is 30 seconds, so allow 35 seconds for BLE
                timeout_seconds = 35.0
            elif command_name == "ping_host":
                # Allow longer for multi-packet pings; default count=4
                count = 4
                if command_args:
                    arg_parts = command_args.split("|")
                    # Format: host|count|port (port optional)
                    if len(arg_parts) >= 2 and arg_parts[1].strip():
                        try:
                            parsed_count = int(arg_parts[1].strip())
                            if parsed_count > 0:
                                count = parsed_count
                        except ValueError:
                            pass
                    elif len(arg_parts) == 1 and arg_parts[0].strip():
                        try:
                            parsed_count = int(arg_parts[0].strip())
                            if parsed_count > 0:
                                count = parsed_count
                        except ValueError:
                            pass
                count = max(1, min(count, 10))
                timeout_seconds = max(10.0, 2.0 + count * 2.0)
            elif command_name == "scanWifi":
                # WiFi scanning can take 5-10 seconds
                timeout_seconds = 15.0
            elif command_name == "rdmp_stream":
                # Establishing the streaming task might take several seconds
                timeout_seconds = 20.0
            elif command_name == "rdmp_stop":
                timeout_seconds = 10.0

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            newline_index = self._notify_buffer.find(b"\n")
            if newline_index != -1:
                raw = self._notify_buffer[:newline_index].decode("utf-8", errors="ignore").strip()
                del self._notify_buffer[: newline_index + 1]
                break
            if loop.time() > deadline:
                # On timeout, return error with any partial data we received
                partial = self._notify_buffer[:200].decode("utf-8", errors="ignore") if self._notify_buffer else ""
                return {"error": f"Bluetooth timeout waiting for response (received: {len(self._notify_buffer)} bytes, partial: {partial[:100]})"}
            await asyncio.sleep(0.01)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues - remove null bytes, extra whitespace
            try:
                cleaned = raw.replace('\x00', '').strip()
                if cleaned != raw:
                    return json.loads(cleaned)
            except:
                pass
            # If still fails, return error with truncated raw response
            return {"error": f"Invalid JSON response: {e}", "raw_response": raw[:200]}

    async def _send_binary_async(self, data: bytes) -> Dict[str, Any]:
        self._ensure_connected()
        assert self.client

        # Clear any stale data before sending binary data
        self._notify_buffer.clear()

        # Write in 20-byte chunks
        mtu = 20
        for i in range(0, len(data), mtu):
            chunk = data[i : i + mtu]
            await self.client.write_gatt_char(self.BINARY_RX_UUID, chunk, response=False)
            await asyncio.sleep(0.005)

        # Expect a JSON line on text TX as acknowledgement
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 5.0
        while True:
            newline_index = self._notify_buffer.find(b"\n")
            if newline_index != -1:
                raw = self._notify_buffer[:newline_index].decode("utf-8", errors="ignore").strip()
                del self._notify_buffer[: newline_index + 1]
                break
            if loop.time() > deadline:
                return {"error": "Bluetooth timeout waiting for binary ack"}
            await asyncio.sleep(0.01)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            try:
                cleaned = raw.replace('\x00', '').strip()
                if cleaned != raw:
                    return json.loads(cleaned)
            except:
                pass
            return {"error": f"Invalid JSON response: {e}", "raw_response": raw[:200]}

    async def _receive_binary_async(self, max_bytes: int) -> bytes:
        if not self._binary_buffer:
            return b""
        data = bytes(self._binary_buffer[:max_bytes])
        del self._binary_buffer[:max_bytes]
        return data

    async def _download_file_async(self, path: str, progress_callback=None) -> bytes:
        # Command
        cmd = f"download_file:{path}\n"
        result = await self._send_command_line_and_collect_stream(cmd, progress_callback)
        return result

    async def _send_command_line_and_collect_stream(self, cmd: str, progress_callback=None) -> bytes:
        self._ensure_connected()
        assert self.client

        # Clear buffers
        self._notify_buffer.clear()

        await self.client.write_gatt_char(self.CHARACTERISTIC_UUID_RX, cmd.encode("utf-8"), response=False)

        # Collect notifications for up to 5 minutes or until JSON closes
        if progress_callback:
            progress_callback(5, "Waiting for stream...")

        response = bytearray()
        start_time = asyncio.get_event_loop().time()
        expected_size = 0

        while (asyncio.get_event_loop().time() - start_time) < 300:
            await asyncio.sleep(0.05)
            if self._notify_buffer:
                # Move all bytes to response buffer
                response += self._notify_buffer
                self._notify_buffer.clear()

                # Lazy parse size
                if expected_size == 0 and b'"size":' in response:
                    try:
                        text_preview = response[:512].decode("utf-8", errors="ignore")
                        if '"size":' in text_preview:
                            idx = text_preview.find('"size":') + 7
                            # read until comma
                            end = text_preview.find(',', idx)
                            if end > idx:
                                expected_size = int(text_preview[idx:end])
                                if progress_callback:
                                    progress_callback(10, f"File size: {expected_size} bytes")
                    except Exception:
                        pass

                if progress_callback and expected_size:
                    # Approximate progress
                    estimated_response_size = int(expected_size * 1.4) + 200
                    progress = min(90, 20 + int(len(response) * 70 / max(1, estimated_response_size)))
                    progress_callback(progress, f"Receiving data... ({len(response)} bytes)")

                # Check for end of JSON: "}
                if len(response) >= 2 and response[-2:] == b'"}':
                    break

        if not response:
            raise ConnectionError("No response from device")

        # Parse JSON wrapper and extract base64 data
        clean_response = response.decode("utf-8", errors="ignore").strip()
        start_idx = clean_response.find('{"status":"success"')
        if start_idx >= 0:
            end_idx = clean_response.rfind('"}') + 2
            if end_idx > start_idx:
                json_part = clean_response[start_idx:end_idx]
            else:
                json_part = clean_response
        else:
            json_part = clean_response

        try:
            data = json.loads(json_part)
        except json.JSONDecodeError as e:
            preview = clean_response[:200] + ("..." if len(clean_response) > 200 else "")
            raise ValueError(f"Invalid JSON response: {e}. Response preview: {preview}")

        if "error" in data:
            raise FileNotFoundError(f"Failed to download file: {data['error']}")
        if "data" not in data:
            raise ValueError(f"Invalid response format: {data}")

        base64_data = data["data"]
        if progress_callback:
            progress_callback(95, "Decoding file...")
        file_bytes = base64.b64decode(base64_data)
        if progress_callback:
            progress_callback(100, "Download complete!")
        return file_bytes

    # ---------- Notify handlers ----------
    def _on_text_notify(self, _handle: int, data: bytearray) -> None:
        # Accumulate; polling loop will detect newlines
        # Defensive check - notification might arrive after loop closure
        try:
            self._notify_buffer += data
        except Exception:
            pass  # Ignore if buffer or loop is invalid

    def _on_binary_notify(self, _handle: int, data: bytearray) -> None:
        # Defensive check - notification might arrive after loop closure
        try:
            self._binary_buffer += data
        except Exception:
            pass  # Ignore if buffer or loop is invalid

    # ---------- Helpers ----------
    def _ensure_connected(self) -> None:
        if not self.is_connected():
            raise ConnectionError("Not connected to Bluetooth device")

    def _start_loop_thread(self) -> None:
        """Start a persistent event loop in a background thread."""
        if self._loop_running and self._loop and not self._loop.is_closed():
            return
        
        self._loop_running = False
        loop_started = threading.Event()
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_running = True
            loop_started.set()
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        loop_started.wait(timeout=2.0)
    
    def _stop_loop_thread(self) -> None:
        """Stop the persistent event loop thread."""
        if self._loop and self._loop_running and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_running = False
            self._loop = None
            if self._loop_thread:
                self._loop_thread.join(timeout=1.0)
            self._loop_thread = None

    def _run(self, coro):
        """Run an async coroutine using persistent event loop or create one."""
        # If we have a persistent loop running, use it
        if self._loop and self._loop_running and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            # Use 60 second timeout to accommodate long-running commands like WiFi connection (35s)
            return future.result(timeout=60)
        
        # Otherwise, try to get existing loop or create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's a running loop, create a new one in a thread
                return self._run_in_thread(coro)
        except RuntimeError:
            pass
        
        # Create a new event loop for this operation
        return asyncio.run(coro)
    
    def _run_in_thread(self, coro):
        """Run a coroutine in a background thread with its own event loop."""
        result = [None]
        exception = [None]
        done = threading.Event()
        
        def run_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                exception[0] = e
            finally:
                new_loop.close()
                done.set()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        done.wait()
        
        if exception[0]:
            raise exception[0]
        return result[0]


