# Copyright (c) 2025 Physical Automation, Inc. All rights reserved.
"""Serial transport implementation for Voxel devices."""

import serial
import json
import time
import base64
from typing import Dict, Any, Optional
from voxel_sdk.voxel import VoxelTransport


class SerialVoxelTransport(VoxelTransport):
    """Serial communication transport for Voxel devices."""
    
    def __init__(self, port: str, baudrate: int = 921600, timeout: float = 5.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection: Optional[serial.Serial] = None
    
    def connect(self, address: str = "") -> None:
        """Connect to serial port. Address parameter is ignored - uses port from constructor."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            # Give device time to initialize
            time.sleep(2)
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from serial port."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.serial_connection = None
    
    def is_connected(self) -> bool:
        """Check if connected to serial port."""
        return self.serial_connection is not None and self.serial_connection.is_open
    
    def send_command(self, command: str, data: Optional[str] = None) -> Dict[str, Any]:
        """Send command over serial and return JSON response."""
        if not self.is_connected():
            raise ConnectionError("Not connected to serial port")
        
        # Build command string
        cmd_str = command
        if data:
            cmd_str += f":{data}"
        cmd_str += "\n"
        
        try:
            # Clear any pending input
            self.serial_connection.reset_input_buffer()
            
            # Send command
            self.serial_connection.write(cmd_str.encode('utf-8'))
            self.serial_connection.flush()
            
            # Small delay to ensure command is processed
            time.sleep(0.1)
            
            # Read response
            response = self.serial_connection.readline().decode('utf-8').strip()
            
            # Parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response}
                
        except serial.SerialException as e:
            return {"error": f"Serial communication error: {e}"}
    
    def send_binary_data(self, data: bytes) -> Dict[str, Any]:
        """Send binary data over serial."""
        if not self.is_connected():
            raise ConnectionError("Not connected to serial port")
        
        try:
            self.serial_connection.write(data)
            self.serial_connection.flush()
            
            # Read response
            response = self.serial_connection.readline().decode('utf-8').strip()
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response", "raw_response": response}
                
        except serial.SerialException as e:
            return {"error": f"Serial communication error: {e}"}
    
    def receive_binary_data(self, max_bytes: int = 1024) -> bytes:
        """Receive binary data from serial."""
        if not self.is_connected():
            raise ConnectionError("Not connected to serial port")
        
        try:
            return self.serial_connection.read(max_bytes)
        except serial.SerialException as e:
            raise ConnectionError(f"Serial communication error: {e}")
    
    def download_file(self, path: str, progress_callback=None) -> bytes:
        """Download a file from the device and return its contents as bytes."""
        if not self.is_connected():
            raise ConnectionError("Not connected to serial port")
        
        # Build command string
        cmd_str = f"download_file:{path}\n"
        
        try:
            # Clear any pending input
            self.serial_connection.reset_input_buffer()
            
            # Send command
            self.serial_connection.write(cmd_str.encode('utf-8'))
            self.serial_connection.flush()
            
            if progress_callback:
                progress_callback(0, "Requesting file...")
            
            # Read response - handle both old and new streaming formats
            response = ""
            start_time = time.time()
            bytes_received = 0
            expected_size = 0
            is_streaming_format = False
            
            # Read initial chunk to determine format
            time.sleep(0.5)  # Give device time to start sending
            initial_chunk = ""
            if self.serial_connection.in_waiting > 0:
                initial_chunk = self.serial_connection.read(min(200, self.serial_connection.in_waiting)).decode('utf-8', errors='ignore')
                response += initial_chunk
                bytes_received += len(initial_chunk)
                
                # Check if this is the new streaming format with JSON header
                if initial_chunk.startswith('{"status":"success"'):
                    is_streaming_format = True
                    if progress_callback:
                        progress_callback(5, "Streaming format detected")
            
            if progress_callback:
                progress_callback(10, f"Receiving data... ({bytes_received} chars)")
            
            # Continue reading based on format
            last_progress_update = start_time
            no_data_count = 0
            while time.time() - start_time < 300:  # 5 minute timeout for large files
                if self.serial_connection.in_waiting > 0:
                    # Read larger chunks for better performance
                    chunk_size = min(4096, self.serial_connection.in_waiting)
                    chunk = self.serial_connection.read(chunk_size).decode('utf-8', errors='ignore')
                    response += chunk
                    bytes_received += len(chunk)
                    
                    # Extract file size if we haven't found it yet
                    if expected_size == 0 and '"size":' in response:
                        try:
                            size_start = response.find('"size":') + 7
                            size_end = response.find(',', size_start)
                            if size_end > size_start:
                                expected_size = int(response[size_start:size_end])
                                if progress_callback:
                                    progress_callback(15, f"File size: {expected_size} bytes")
                        except:
                            pass
                    
                    # Update progress less frequently for better performance
                    current_time = time.time()
                    if progress_callback and (current_time - last_progress_update) > 0.5:  # Update every 0.5 seconds
                        if expected_size > 0:
                            # Estimate progress based on base64 size
                            estimated_response_size = int(expected_size * 1.4) + 200
                            progress = min(90, 20 + int(bytes_received * 70 / estimated_response_size))
                        else:
                            # Fallback progress based on data received
                            progress = min(50, 15 + bytes_received // 10000)
                        
                        progress_callback(progress, f"Receiving data... ({bytes_received} chars)")
                        last_progress_update = current_time
                    
                    # Check if response is complete
                    if is_streaming_format and ('"}' in response[-10:]):  # Check last 10 chars for closing
                        break
                    elif not is_streaming_format:
                        # For old format, check if we have a complete JSON structure
                        if response.startswith('{"status":"success"') and '"data":"' in response:
                            # Look for closing pattern (might have \r\n after)
                            if '"}' in response[-20:]:  # Check last 20 chars for closing
                                break
                        elif len(response) > 100 and not self.serial_connection.in_waiting:
                            # Wait a bit to see if more data comes
                            time.sleep(0.2)
                            if not self.serial_connection.in_waiting:
                                break
                        
                else:
                    # No data available, sleep briefly
                    no_data_count += 1
                    time.sleep(0.005)  # Reduced sleep time for better responsiveness
                    
                    # If we haven't seen data for a while and have some response, check if it's complete
                    if no_data_count > 100 and len(response) > 50:  # 0.5 seconds of no data
                        if response.startswith('{"status":"success"') and '"}' in response:
                            if progress_callback:
                                progress_callback(95, "Response appears complete, processing...")
                            break
            
            if not response:
                raise ConnectionError("No response from device")
            
            # Parse response based on format
            if is_streaming_format or response.startswith('{"status":"success"'):
                # New streaming format with JSON wrapper
                try:
                    # Clean up the response - remove extra whitespace and line endings
                    clean_response = response.strip()
                    
                    # Find the JSON part (from { to })
                    start_idx = clean_response.find('{"status":"success"')
                    if start_idx >= 0:
                        end_idx = clean_response.rfind('"}') + 2
                        if end_idx > start_idx:
                            json_part = clean_response[start_idx:end_idx]
                            data = json.loads(json_part)
                        else:
                            data = json.loads(clean_response)
                    else:
                        data = json.loads(clean_response)
                    
                    if "error" in data:
                        raise FileNotFoundError(f"Failed to download file {path}: {data['error']}")
                    if "data" not in data:
                        raise ValueError(f"Invalid response format: {data}")
                    base64_data = data["data"]
                except json.JSONDecodeError as e:
                    preview = response[:200] + "..." if len(response) > 200 else response
                    raise ValueError(f"Invalid JSON response: {e}. Response preview: {preview}")
            else:
                # Old format - raw base64 data
                if progress_callback:
                    progress_callback(85, "Processing raw base64 data...")
                
                # The entire response should be base64 data
                base64_data = response.strip()
                
                # Basic validation and cleanup for base64
                if not base64_data:
                    raise ValueError("No base64 data received")
                
                # Clean up base64 data - remove any non-base64 characters
                import re
                base64_data = re.sub(r'[^A-Za-z0-9+/=]', '', base64_data)
                
                # Ensure proper base64 padding
                missing_padding = len(base64_data) % 4
                if missing_padding:
                    base64_data += '=' * (4 - missing_padding)
                
                if not base64_data:
                    preview = response[:200] + "..." if len(response) > 200 else response
                    raise ValueError(f"No valid base64 data found. Preview: {preview}")
            
            if progress_callback:
                progress_callback(90, "Decoding file...")
            
            # Decode base64 data
            try:
                file_data = base64.b64decode(base64_data)
                if progress_callback:
                    progress_callback(100, "Download complete!")
                return file_data
            except Exception as e:
                raise ValueError(f"Failed to decode file data: {e}")
                
        except serial.SerialException as e:
            raise ConnectionError(f"Serial communication error: {e}")