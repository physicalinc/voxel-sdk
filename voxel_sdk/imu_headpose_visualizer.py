#!/usr/bin/env python3
"""
IMU head-pose visualization for Voxel devices - Web interface.

Starts a Flask web server that provides a real-time 3D head pose visualization
driven by IMU data from the device. The visualization uses Three.js for smooth
3D rendering in the browser.

Example
-------
python -m voxel_sdk.imu_headpose_visualizer --port /dev/tty.usbserial-1410

Connectivity
------------
Select the transport explicitly with ``--transport serial`` (requires
``--port``) or ``--transport ble`` (optionally with ``--ble-name`` /
``--ble-address``). With ``--transport auto`` (default) the script picks serial
when a port is provided, otherwise it interactively prompts for serial vs BLE.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

# Support both package (`python -m`) and script (`python path/to/...py`) execution.
import os

# Get the directory containing voxel_sdk (voxel-sdk/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # voxel_sdk/
PACKAGE_PARENT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # voxel-sdk/

if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

# Use absolute imports - all SDK modules now use absolute imports too
from voxel_sdk.device_controller import DeviceController  # noqa: E402
from voxel_sdk.serial import SerialVoxelTransport  # noqa: E402

try:
    from flask import Flask, jsonify, render_template_string, Response
    from flask_cors import CORS
except ImportError:
    print("Error: Flask and flask-cors are required for web visualization.")
    print("Install with: pip install flask flask-cors")
    sys.exit(1)


def _import_ble_transport():
    from voxel_sdk.ble import BleVoxelTransport  # noqa: E402
    return BleVoxelTransport


# ---------------------------------------------------------------------------
# Data modelling
# ---------------------------------------------------------------------------


@dataclass
class ImuSample:
    """Normalized IMU sample pulled from the device."""

    timestamp_ms: int
    accel: np.ndarray  # m/s^2
    gyro: np.ndarray  # rad/s
    mag: np.ndarray  # microtesla

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "accel": {"x": float(self.accel[0]), "y": float(self.accel[1]), "z": float(self.accel[2])},
            "gyro": {"x": float(self.gyro[0]), "y": float(self.gyro[1]), "z": float(self.gyro[2])},
            "mag": {"x": float(self.mag[0]), "y": float(self.mag[1]), "z": float(self.mag[2])},
        }


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / norm
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Construct quaternion following aerospace convention (XYZ intrinsic)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def quat_to_euler_degrees(q: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees."""
    w, x, y, z = quat_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.degrees(math.copysign(math.pi / 2, sinp))
    else:
        pitch = math.degrees(math.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return quat_normalize(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize((s0 * q0) + (s1 * q1))


# ---------------------------------------------------------------------------
# Orientation filter
# ---------------------------------------------------------------------------


class ComplementaryOrientationFilter:
    """Fuse gyroscope integration with accel/mag absolute reference."""

    def __init__(self, gyro_weight: float = 0.96) -> None:
        if not (0.0 < gyro_weight <= 1.0):
            raise ValueError("gyro_weight must be in (0, 1]")
        self.gyro_weight = gyro_weight
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.last_timestamp_ms: Optional[int] = None

    def reset(self) -> None:
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.last_timestamp_ms = None

    def update(self, sample: ImuSample) -> np.ndarray:
        dt = self._compute_dt(sample.timestamp_ms)
        gyro_quat = self._integrate_gyro(sample.gyro, dt)
        predicted = quat_multiply(self.quaternion, gyro_quat)

        reference = self._estimate_from_accel_mag(sample.accel, sample.mag)
        if reference is None:
            fused = predicted
        else:
            fused = quat_slerp(reference, predicted, self.gyro_weight)

        self.quaternion = quat_normalize(fused)
        return self.quaternion

    def _compute_dt(self, timestamp_ms: int) -> float:
        if self.last_timestamp_ms is None:
            self.last_timestamp_ms = timestamp_ms
            return 0.1  # default to sensor cadence
        dt = (timestamp_ms - self.last_timestamp_ms) / 1000.0
        self.last_timestamp_ms = timestamp_ms
        if dt <= 0.0 or dt > 1.0:
            return 0.1
        return dt

    @staticmethod
    def _integrate_gyro(gyro: np.ndarray, dt: float) -> np.ndarray:
        omega = np.asarray(gyro, dtype=float)
        angle = np.linalg.norm(omega) * dt
        if angle == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        axis = omega / np.linalg.norm(omega)
        return quat_from_axis_angle(axis, angle)

    @staticmethod
    def _estimate_from_accel_mag(accel: np.ndarray, mag: np.ndarray) -> Optional[np.ndarray]:
        accel = np.asarray(accel, dtype=float)
        a_norm = np.linalg.norm(accel)
        if a_norm < 1e-6:
            return None
        ax, ay, az = accel / a_norm

        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        mag = np.asarray(mag, dtype=float)
        m_norm = np.linalg.norm(mag)
        if m_norm < 1e-6:
            yaw = 0.0
        else:
            mx, my, mz = mag / m_norm
            mx_comp = mx * math.cos(pitch) + mz * math.sin(pitch)
            my_comp = (
                mx * math.sin(roll) * math.sin(pitch)
                + my * math.cos(roll)
                - mz * math.sin(roll) * math.cos(pitch)
            )
            yaw = math.atan2(-my_comp, mx_comp)

        return quat_from_euler(roll, pitch, yaw)


# ---------------------------------------------------------------------------
# IMU client
# ---------------------------------------------------------------------------


class DeviceImuClient:
    """Thin wrapper over DeviceController.imu_capture with retries."""

    def __init__(self, controller: DeviceController, retries: int = 3) -> None:
        self.controller = controller
        self.retries = max(1, retries)

    def fetch_sample(self) -> Optional[ImuSample]:
        last_error: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                response = self.controller.imu_capture()
                if isinstance(response, dict):
                    if response.get("status") != "success":
                        last_error = RuntimeError(str(response))
                        continue
                    payload = response.get("data")
                    if not isinstance(payload, dict):
                        last_error = RuntimeError(f"Unexpected payload: {payload}")
                        continue

                    return ImuSample(
                        timestamp_ms=int(payload.get("timestamp_ms", 0)),
                        accel=np.array(
                            [
                                float(payload["accel"]["x"]),
                                float(payload["accel"]["y"]),
                                float(payload["accel"]["z"]),
                            ]
                        ),
                        gyro=np.array(
                            [
                                float(payload["gyro"]["x"]),
                                float(payload["gyro"]["y"]),
                                float(payload["gyro"]["z"]),
                            ]
                        ),
                        mag=np.array(
                            [
                                float(payload["mag"]["x"]),
                                float(payload["mag"]["y"]),
                                float(payload["mag"]["z"]),
                            ]
                        ),
                    )
                last_error = RuntimeError(f"Unexpected response type: {type(response)}")
            except KeyError as exc:
                last_error = RuntimeError(f"Missing field in IMU response: {exc}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.05)

        if last_error:
            print(f"[imu_headpose_visualizer] Failed to fetch IMU sample: {last_error}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------


class HeadPoseWebServer:
    """Flask web server for head pose visualization."""

    def __init__(self, imu_client: DeviceImuClient, filter_: ComplementaryOrientationFilter, port: int = 5000):
        self.imu_client = imu_client
        self.filter = filter_
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self.latest_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.latest_sample: Optional[ImuSample] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        @self.app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/imu")
        def get_imu():
            """Get latest IMU data and orientation."""
            roll, pitch, yaw = quat_to_euler_degrees(self.latest_quaternion)
            quat_list = self.latest_quaternion.tolist()
            return jsonify(
                {
                    "quaternion": {"w": quat_list[0], "x": quat_list[1], "y": quat_list[2], "z": quat_list[3]},
                    "euler": {"roll": roll, "pitch": pitch, "yaw": yaw},
                    "imu": self.latest_sample.to_dict() if self.latest_sample else None,
                }
            )

    def _update_loop(self):
        """Background thread that continuously polls IMU and updates orientation."""
        while self.running:
            sample = self.imu_client.fetch_sample()
            if sample is not None:
                self.latest_quaternion = self.filter.update(sample)
                self.latest_sample = sample
            time.sleep(0.05)  # ~20 Hz update rate

    def start(self):
        """Start the web server and background update thread."""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"\n🌐 Head pose visualization server starting...")
        print(f"   Open your browser to: http://localhost:{self.port}")
        print(f"   Press Ctrl+C to stop\n")
        self.app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)

    def stop(self):
        """Stop the server and background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# HTML template with Three.js visualization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voxel Head Pose Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #000000;
            color: #e3e7ff;
            overflow: hidden;
        }
        #container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        #canvas {
            display: block;
        }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.8;
            min-width: 280px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        #info h2 {
            margin-bottom: 12px;
            font-size: 18px;
            color: #5a86ff;
            font-weight: 600;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }
        .stat-label {
            color: #8b92b8;
        }
        .stat-value {
            color: #e3e7ff;
            font-weight: 500;
        }
        #status {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 12px 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 12px;
            color: #69f0ae;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #69f0ae;
            margin-right: 8px;
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div id="container">
        <canvas id="canvas"></canvas>
        <div id="info">
            <h2>Head Pose</h2>
            <div class="stat">
                <span class="stat-label">Roll:</span>
                <span class="stat-value" id="roll">0.0°</span>
            </div>
            <div class="stat">
                <span class="stat-label">Pitch:</span>
                <span class="stat-value" id="pitch">0.0°</span>
            </div>
            <div class="stat">
                <span class="stat-label">Yaw:</span>
                <span class="stat-value" id="yaw">0.0°</span>
            </div>
            <div class="stat" style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
                <span class="stat-label">Timestamp:</span>
                <span class="stat-value" id="timestamp">0 ms</span>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
                <h3 style="font-size: 14px; color: #5a86ff; font-weight: 600; margin-bottom: 8px;">Acceleration (m/s²)</h3>
                <div class="stat">
                    <span class="stat-label">X:</span>
                    <span class="stat-value" id="accel-x">0.00</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Y:</span>
                    <span class="stat-value" id="accel-y">0.00</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Z:</span>
                    <span class="stat-value" id="accel-z">0.00</span>
                </div>
                <div class="stat" style="margin-top: 4px;">
                    <span class="stat-label">Magnitude:</span>
                    <span class="stat-value" id="accel-mag">0.00</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 11px; color: #8b92b8;">
                <div style="margin-bottom: 6px;"><strong style="color: #e3e7ff;">Axes:</strong></div>
                <div style="margin-bottom: 4px;">
                    <span style="color: #ff0000;">●</span> IMU X/Y/Z (rotates with head)
                </div>
            </div>
        </div>
        <div id="status">
            <span class="status-dot"></span>
            <span id="status-text">Connected</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene setup - pure black background
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0.15, 0.6);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas'), antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);

        // Simple lighting for particles
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 1.0, 10);
        pointLight.position.set(0, 0.5, 1);
        scene.add(pointLight);

        // Create head group for IMU axes
        const headGroup = new THREE.Group();

        // IMU coordinate system axes - attached to head, rotate with IMU
        const imuAxisLength = 0.12;
        const imuAxisRadius = 0.006;
        
        // IMU X-axis (Red) - in IMU's local coordinate system
        const imuXAxisGeometry = new THREE.CylinderGeometry(imuAxisRadius, imuAxisRadius, imuAxisLength, 8);
        imuXAxisGeometry.rotateZ(Math.PI / 2);
        const imuXAxisMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000, emissive: 0xff0000 });
        const imuXAxis = new THREE.Mesh(imuXAxisGeometry, imuXAxisMaterial);
        imuXAxis.position.set(imuAxisLength / 2, 0, 0);
        headGroup.add(imuXAxis);
        
        const imuXArrow = new THREE.ConeGeometry(imuAxisRadius * 2, imuAxisRadius * 3, 8);
        imuXArrow.rotateZ(Math.PI / 2);
        const imuXArrowMesh = new THREE.Mesh(imuXArrow, imuXAxisMaterial);
        imuXArrowMesh.position.set(imuAxisLength, 0, 0);
        headGroup.add(imuXArrowMesh);
        
        // IMU Y-axis (Green)
        const imuYAxisGeometry = new THREE.CylinderGeometry(imuAxisRadius, imuAxisRadius, imuAxisLength, 8);
        const imuYAxisMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, emissive: 0x00ff00 });
        const imuYAxis = new THREE.Mesh(imuYAxisGeometry, imuYAxisMaterial);
        imuYAxis.position.set(0, imuAxisLength / 2, 0);
        headGroup.add(imuYAxis);
        
        const imuYArrow = new THREE.ConeGeometry(imuAxisRadius * 2, imuAxisRadius * 3, 8);
        const imuYArrowMesh = new THREE.Mesh(imuYArrow, imuYAxisMaterial);
        imuYArrowMesh.position.set(0, imuAxisLength, 0);
        headGroup.add(imuYArrowMesh);
        
        // IMU Z-axis (Blue)
        const imuZAxisGeometry = new THREE.CylinderGeometry(imuAxisRadius, imuAxisRadius, imuAxisLength, 8);
        imuZAxisGeometry.rotateX(Math.PI / 2);
        const imuZAxisMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff, emissive: 0x0000ff });
        const imuZAxis = new THREE.Mesh(imuZAxisGeometry, imuZAxisMaterial);
        imuZAxis.position.set(0, 0, imuAxisLength / 2);
        headGroup.add(imuZAxis);
        
        const imuZArrow = new THREE.ConeGeometry(imuAxisRadius * 2, imuAxisRadius * 3, 8);
        imuZArrow.rotateX(-Math.PI / 2);
        const imuZArrowMesh = new THREE.Mesh(imuZArrow, imuZAxisMaterial);
        imuZArrowMesh.position.set(0, 0, imuAxisLength);
        headGroup.add(imuZArrowMesh);
        
        // IMU axis labels
        function createIMULabel(text, color) {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const context = canvas.getContext('2d');
            context.fillStyle = 'rgba(0, 0, 0, 0)';
            context.fillRect(0, 0, 128, 128);
            context.font = 'Bold 60px Arial';
            context.fillStyle = '#' + color.toString(16).padStart(6, '0');
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(text, 64, 64);
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(0.04, 0.04, 1);
            return sprite;
        }
        
        const imuXLabel = createIMULabel('X', 0xff0000);
        imuXLabel.position.set(imuAxisLength + 0.025, 0, 0);
        headGroup.add(imuXLabel);
        
        const imuYLabel = createIMULabel('Y', 0x00ff00);
        imuYLabel.position.set(0, imuAxisLength + 0.025, 0);
        headGroup.add(imuYLabel);
        
        const imuZLabel = createIMULabel('Z', 0x0000ff);
        imuZLabel.position.set(0, 0, imuAxisLength + 0.025);
        headGroup.add(imuZLabel);

        scene.add(headGroup);

        // Subtle grid (very faint on black)
        const gridHelper = new THREE.GridHelper(0.6, 20, 0x111111, 0x080808);
        gridHelper.position.y = -0.25;
        scene.add(gridHelper);

        // Motion trail (bright on black)
        const trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.LineBasicMaterial({ 
            color: 0xffffff, 
            linewidth: 2, 
            opacity: 0.5, 
            transparent: true 
        });
        const trail = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trail);

        const trailPoints = [];
        const maxTrailLength = 120;

        // Quaternion for head rotation
        const quaternion = new THREE.Quaternion();

        // Update function
        function updatePose(data) {
            if (!data || !data.quaternion) return;

            // Update quaternion
            quaternion.set(data.quaternion.x, data.quaternion.y, data.quaternion.z, data.quaternion.w);
            headGroup.setRotationFromQuaternion(quaternion);

            // Update trail (IMU Z-axis tip)
            const frontPoint = new THREE.Vector3(0, 0, imuAxisLength);
            frontPoint.applyQuaternion(quaternion);
            trailPoints.push(frontPoint.clone());
            if (trailPoints.length > maxTrailLength) {
                trailPoints.shift();
            }
            if (trailPoints.length > 1) {
                trailGeometry.setFromPoints(trailPoints);
            }

            // Update UI
            if (data.euler) {
                document.getElementById('roll').textContent = data.euler.roll.toFixed(1) + '°';
                document.getElementById('pitch').textContent = data.euler.pitch.toFixed(1) + '°';
                document.getElementById('yaw').textContent = data.euler.yaw.toFixed(1) + '°';
            }
            if (data.imu) {
                document.getElementById('timestamp').textContent = data.imu.timestamp_ms + ' ms';
                
                // Update acceleration display
                if (data.imu.accel) {
                    const accelX = data.imu.accel.x;
                    const accelY = data.imu.accel.y;
                    const accelZ = data.imu.accel.z;
                    const accelMag = Math.sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);
                    
                    document.getElementById('accel-x').textContent = accelX.toFixed(2);
                    document.getElementById('accel-y').textContent = accelY.toFixed(2);
                    document.getElementById('accel-z').textContent = accelZ.toFixed(2);
                    document.getElementById('accel-mag').textContent = accelMag.toFixed(2);
                }
            }
        }

        // Fetch IMU data
        let lastFetch = 0;
        async function fetchIMU() {
            try {
                const response = await fetch('/api/imu');
                const data = await response.json();
                updatePose(data);
                document.getElementById('status-text').textContent = 'Connected';
                lastFetch = Date.now();
            } catch (error) {
                console.error('Error fetching IMU data:', error);
                document.getElementById('status-text').textContent = 'Disconnected';
            }
        }

        // Poll for updates
        setInterval(fetchIMU, 50); // 20 Hz
        fetchIMU(); // Initial fetch

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_transport_choice(args: argparse.Namespace) -> str:
    """Determine transport type from args or prompt user."""
    if args.transport != "auto":
        return args.transport

    if args.port:
        return "serial"

    print("\nSelect connection method:")
    print("  1. Serial (wired)")
    print("  2. Bluetooth (BLE)")
    choice = input("Enter choice [1-2]: ").strip()
    return "serial" if choice == "1" else "ble"


def connect_controller(args: argparse.Namespace) -> tuple[DeviceController, str]:
    """Connect to device and return controller."""
    transport_type = resolve_transport_choice(args)

    if transport_type == "serial":
        if not args.port:
            print("Error: --port is required for serial transport")
            sys.exit(1)
        transport = SerialVoxelTransport(port=args.port, baudrate=args.baudrate, timeout=args.timeout)
        controller = DeviceController(transport)
        controller.connect()
        return controller, "serial"

    # BLE
    BleVoxelTransport = _import_ble_transport()
    device_name = args.ble_name or "voxel"
    transport = BleVoxelTransport(device_name=device_name)
    controller = DeviceController(transport)
    controller.connect(args.ble_address or "")
    return controller, "ble"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxel-sdk head pose visualizer",
        description="Start a web server for 3D head pose visualization driven by IMU data.",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "serial", "ble"],
        default="auto",
        help="Transport type: auto, serial, or ble (default: auto)",
    )
    parser.add_argument(
        "--port",
        help="Serial port (e.g. /dev/ttyUSB0). Required for serial transport.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=921600,
        help="Serial baudrate (default: 921600).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Serial read timeout in seconds (default: 5).",
    )
    parser.add_argument(
        "--ble-name",
        default="voxel",
        help="BLE device name prefix to search for (default: voxel).",
    )
    parser.add_argument(
        "--ble-address",
        default="",
        help="Optional explicit BLE device address (bypasses scanning).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=5000,
        help="Web server port (default: 5000).",
    )
    parser.add_argument(
        "--gyro-weight",
        type=float,
        default=0.96,
        help="Complementary filter weight for gyro prediction (0-1, default: 0.96).",
    )
    return parser


def install_signal_handlers(server: HeadPoseWebServer, controller: DeviceController) -> None:
    def _cleanup_handler(_signum, _frame):
        try:
            server.stop()
            controller.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        controller, transport_type = connect_controller(args)
        print(f"✓ Connected via {transport_type}")
    except Exception as exc:  # noqa: BLE001
        print(f"[imu_headpose_visualizer] Failed to connect: {exc}", file=sys.stderr)
        return 1

    imu_client = DeviceImuClient(controller)
    orientation_filter = ComplementaryOrientationFilter(gyro_weight=args.gyro_weight)

    server = HeadPoseWebServer(imu_client, orientation_filter, port=args.web_port)
    install_signal_handlers(server, controller)

    try:
        server.start()
    finally:
        server.stop()
        try:
            controller.disconnect()
        except Exception:  # noqa: BLE001
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
