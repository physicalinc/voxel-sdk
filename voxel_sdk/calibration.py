# Copyright (c) 2025 Physical Automation, Inc.
"""Camera intrinsic calibration utilities using OpenCV.

This module displays an on-screen calibration pattern (chessboard), streams frames
from the device, detects the pattern from multiple viewpoints, computes intrinsics,
and saves them back to the device as a JSON text file.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
	# Optional dependencies
	import cv2  # type: ignore
	import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
	cv2 = None  # type: ignore
	np = None  # type: ignore


@dataclass
class CalibrationResult:
	label: str
	image_size: Tuple[int, int]
	camera_matrix: List[List[float]]
	dist_coeffs: List[float]
	rms: float
	num_samples: int
	device_path: str
	timestamp: str


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _choose_local_ip() -> str:
	"""Pick a usable local IP address for device to connect back."""
	def valid(ip: str) -> bool:
		return ip and not ip.startswith("127.") and ip != "0.0.0.0"

	try:
		with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as temp:
			temp.connect(("8.8.8.8", 80))
			candidate = temp.getsockname()[0]
			if valid(candidate):
				return candidate
	except Exception:
		pass

	try:
		hostname_ips = socket.gethostbyname_ex(socket.gethostname())[2]
		for ip in hostname_ips:
			if valid(ip):
				return ip
	except Exception:
		pass

	raise RuntimeError(
		"Unable to determine local IP address for calibration stream. "
		"Specify a network interface or ensure network is available."
	)


def _gen_chessboard(board_cols: int, board_rows: int, square_px: int = 80, margin_px: int = 60, invert: bool = False) -> "np.ndarray":
	"""Generate a chessboard image (board_cols x board_rows inner corners)."""
	if np is None or cv2 is None:
		raise RuntimeError("OpenCV and numpy are required for calibration. Install with `pip install opencv-python numpy`.")

	# The number of squares is inner corners + 1 in each direction
	squares_x = board_cols + 1
	squares_y = board_rows + 1
	board_w = squares_x * square_px
	board_h = squares_y * square_px
	img_w = board_w + margin_px * 2
	img_h = board_h + margin_px * 2

	bg = 255 if invert else 0
	fg = 0 if invert else 255
	image = np.full((img_h, img_w, 3), bg, dtype=np.uint8)

	for y in range(squares_y):
		for x in range(squares_x):
			if (x + y) % 2 == 0:
				x0 = margin_px + x * square_px
				y0 = margin_px + y * square_px
				cv2.rectangle(image, (x0, y0), (x0 + square_px, y0 + square_px), (fg, fg, fg), thickness=-1)

	text = f"Voxel Calibration {board_cols}x{board_rows}"
	cv2.putText(
		image,
		text,
		(margin_px, img_h - int(margin_px * 0.5)),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(128, 128, 128),
		2,
		cv2.LINE_AA,
	)
	return image


def _open_fullscreen(window_name: str) -> None:
	if cv2 is None:
		return
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	# Try to force fullscreen if platform allows
	try:
		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	except Exception:
		pass


def _recv_exact(conn: socket.socket, length: int) -> Optional[bytes]:
	data = bytearray()
	while len(data) < length:
		chunk = conn.recv(length - len(data))
		if not chunk:
			return None
		data.extend(chunk)
	return bytes(data)


def _find_free_port(base_port: int = 9300, max_tries: int = 50) -> int:
	for i in range(max_tries):
		port = base_port + i
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				s.bind(("", port))
				return port
			except Exception:
				continue
	raise RuntimeError("Unable to find a free TCP port for calibration stream.")


def _format_intrinsics_json(label: str, image_size: Tuple[int, int], camera_matrix: "np.ndarray", dist_coeffs: "np.ndarray", rms: float, samples: int) -> str:
	w, h = image_size
	payload: Dict[str, Any] = {
		"label": label,
		"image_width": int(w),
		"image_height": int(h),
		"camera_matrix": camera_matrix.tolist(),
		"dist_coeffs": dist_coeffs.reshape(-1).tolist(),
		"rms": float(rms),
		"samples": int(samples),
		"timestamp": _utc_now_iso(),
		"type": "voxel_intrinsics_v1",
	}
	# Compact JSON to avoid newlines in command payloads (transport is line-based)
	return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def calibrate_and_save(
	controller: Any,
	label: str = "",
	requested_samples: int = 0,
	board_cols: int = 9,
	board_rows: int = 6,
	min_samples_default: int = 15,
	start_with_preview: bool = True,
	connect_timeout_sec: float = 20.0,
) -> CalibrationResult:
	"""Run intrinsics calibration for a single device and save intrinsics JSON on-device.

	Returns a CalibrationResult with metadata and the device file path.
	"""
	if cv2 is None or np is None:
		raise RuntimeError("OpenCV and numpy are required. Install with `pip install opencv-python numpy`.")

	filesystem = controller.filesystem  # DeviceController expected

	# Prepare calibration target (use a larger margin to host a non-occluding PiP preview)
	local_margin_px = 160
	square_px = 90
	# Default to inverted (white background with black squares) for better visibility
	invert_pattern = True
	pattern = _gen_chessboard(board_cols, board_rows, square_px=square_px, margin_px=local_margin_px, invert=invert_pattern)
	target_window = "Voxel Calibration Target"
	_open_fullscreen(target_window)
	cv2.imshow(target_window, pattern)
	cv2.waitKey(1)

	# Prepare stream listener
	listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	port = _find_free_port(9300)
	listener.bind(("", port))
	listener.listen(1)

	# Ask device to connect back
	target_host = _choose_local_ip()
	print(f"[calibrate] Device will connect to {target_host}:{port}")
	
	# Optional: Test connectivity first (non-blocking, just for diagnostics)
	try:
		ping_resp = filesystem.ping_host(target_host, count=1)
		if isinstance(ping_resp, dict) and ping_resp.get("status") != "success":
			print(f"[calibrate] Warning: Device ping to {target_host} failed - connectivity may be an issue")
	except Exception:
		# Ping not critical, continue anyway
		pass
	
	response = filesystem.start_rdmp_stream(target_host, port, quality=None)
	if isinstance(response, dict) and "error" in response:
		listener.close()
		cv2.destroyWindow(target_window)
		raise RuntimeError(
			f"Device failed to start streaming: {response}\n"
			f"Device tried to connect to {target_host}:{port} but connection failed.\n"
			f"Check that the device can reach this IP address on your network."
		)

	# Non-blocking accept loop that continues to refresh the UI to avoid gray screen
	listener.settimeout(0.1)
	start_wait = time.time()
	conn = None  # type: ignore[assignment]
	dots = 0
	while True:
		try:
			conn, _addr = listener.accept()
			break
		except socket.timeout:
			# Update waiting screen
			wait_vis = pattern.copy()
			msg = f"Waiting for device to connect {target_host}:{port}" + "." * (dots % 4)
			dots += 1
			cv2.putText(wait_vis, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
			cv2.putText(wait_vis, "If prompted, allow incoming connections (firewall).", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
			cv2.putText(wait_vis, "Press 'q' to cancel.", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.imshow(target_window, wait_vis)
			key = cv2.waitKey(1)
			if key == ord("q") or key == 27:
				listener.close()
				try:
					filesystem.stop_rdmp_stream()
				except Exception:
					pass
				raise RuntimeError("Calibration cancelled by user")
			if (time.time() - start_wait) > connect_timeout_sec:
				listener.close()
				cv2.destroyWindow(target_window)
				try:
					filesystem.stop_rdmp_stream()
				except Exception:
					pass
				raise TimeoutError(
					f"Timed out waiting for device to connect for calibration stream.\n"
					f"Device was told to connect to {target_host}:{port} but never connected.\n"
					f"Possible issues:\n"
					f"  1. macOS Firewall is blocking incoming connections - check System Settings > Network > Firewall\n"
					f"  2. Device and computer are on different networks/subnets\n"
					f"  3. Device cannot reach IP address {target_host}\n"
					f"Try: ping {target_host} from another device on the same network to verify reachability."
				)
		except Exception:
			# Any other accept error: clean up and raise
			listener.close()
			cv2.destroyWindow(target_window)
			try:
				filesystem.stop_rdmp_stream()
			except Exception:
				pass
			raise
	# Connected; close the listener socket
	listener.close()

	conn.settimeout(5.0)

	# Detection and collection loop
	show_preview = bool(start_with_preview)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	inner = (board_cols, board_rows)

	objp = np.zeros((board_rows * board_cols, 3), np.float32)
	objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)  # square size = 1.0 unit

	objpoints: List["np.ndarray"] = []
	imgpoints: List["np.ndarray"] = []
	last_center: Optional[Tuple[float, float]] = None
	last_area: Optional[float] = None
	samples_needed = requested_samples if requested_samples > 0 else min_samples_default
	last_accept_time = 0.0
	frames_seen = 0
	status_flash_msg: Optional[str] = None
	status_flash_until: float = 0.0

	try:
		while True:
			header = _recv_exact(conn, 8)
			if not header:
				break
			if header[:4] != b"VXL0":
				break
			frame_len = struct.unpack(">I", header[4:])[0]
			if frame_len <= 0 or frame_len > 5 * 1024 * 1024:
				break
			payload = _recv_exact(conn, frame_len)
			if not payload:
				break
			frame_array = np.frombuffer(payload, dtype=np.uint8)
			image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
			if image is None:
				continue
			frames_seen += 1

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			found, corners = cv2.findChessboardCorners(gray, inner, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

			accepted_now = False
			if found and corners is not None:
				corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
				cv2.drawChessboardCorners(image, inner, corners2, found)

				center_now = tuple(corners2.mean(axis=0).ravel().tolist())
				# Estimate current pattern area using bounding box
				xs = corners2[:, 0, 0]
				ys = corners2[:, 0, 1]
				width_now = float(xs.max() - xs.min())
				height_now = float(ys.max() - ys.min())
				area_now = max(1.0, width_now * height_now)
				now = time.time()
				# Accept if sufficient motion since last or time gap
				accept = False
				if last_center is None:
					accept = True
				else:
					dx = center_now[0] - last_center[0]
					dy = center_now[1] - last_center[1]
					dist = (dx * dx + dy * dy) ** 0.5
					scale_ok = False
					if last_area is not None:
						scale_ratio = abs(area_now - last_area) / max(last_area, 1.0)
						scale_ok = scale_ratio > 0.12  # >12% change in apparent size
					accept = dist > 30.0 or scale_ok or (now - last_accept_time) > 1.2

				if accept:
					objpoints.append(objp.copy())
					imgpoints.append(corners2)
					last_center = center_now
					last_area = area_now
					last_accept_time = now
					accepted_now = True
					# Console feedback for reassurance
					print(f"[calibrate] Captured sample {len(imgpoints)}/{samples_needed}")

			# HUD overlay
			cv2.putText(image, f"Samples: {len(imgpoints)}/{samples_needed}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(image, f"Pattern: {'FOUND' if found else 'NOT FOUND'}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if found else (0, 0, 255), 2, cv2.LINE_AA)
			cv2.putText(image, "Press 'c' capture, 'p' preview on/off, 'q' finish", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

			# Always update target window with status overlay on top of the grid
			pattern_vis = pattern.copy()
			cv2.putText(pattern_vis, f"Samples: {len(imgpoints)}/{samples_needed}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(pattern_vis, f"Pattern: {'FOUND' if found else 'NOT FOUND'}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if found else (0, 0, 255), 2, cv2.LINE_AA)
			if accepted_now:
				cv2.putText(pattern_vis, "Sample captured ✓", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(pattern_vis, "Keys: p=preview c=capture i=invert q=finish", (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
			# Flash temporary status messages
			if status_flash_msg and time.time() < status_flash_until:
				cv2.putText(pattern_vis, status_flash_msg, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
			else:
				status_flash_msg = None
			# Picture-in-picture preview in bottom-right margin so it doesn't cover the chessboard
			if show_preview:
				try:
					h_img, w_img = image.shape[:2]
					# Fit inside the right/bottom margin
					max_w = max(120, local_margin_px - 20)
					target_w = min(320, max_w)
					scale = target_w / float(w_img)
					target_h = int(h_img * scale)
					# Ensure height also fits in margin
					if target_h > (local_margin_px - 20):
						scale = (local_margin_px - 20) / float(h_img)
						target_w = int(w_img * scale)
						target_h = int(h_img * scale)
					if target_w > 20 and target_h > 20:
						preview = cv2.resize(image, (target_w, target_h))
						x0 = pattern_vis.shape[1] - local_margin_px + (local_margin_px - target_w) // 2
						y0 = pattern_vis.shape[0] - local_margin_px + (local_margin_px - target_h) // 2
						# Draw a subtle border
						cv2.rectangle(pattern_vis, (x0 - 4, y0 - 4), (x0 + target_w + 4, y0 + target_h + 4), (30, 30, 30), thickness=-1)
						# Paste preview
						pattern_vis[y0 : y0 + target_h, x0 : x0 + target_w] = preview
						cv2.rectangle(pattern_vis, (x0 - 4, y0 - 4), (x0 + target_w + 4, y0 + target_h + 4), (200, 200, 200), thickness=1)
				except Exception:
					pass
			cv2.imshow(target_window, pattern_vis)
			key = cv2.waitKey(1)
			if key == ord("q") or key == 27:
				break
			if key == ord("c"):
				if found and corners is not None:
					objpoints.append(objp.copy())
					imgpoints.append(corners2 if 'corners2' in locals() else corners)  # use refined if available
					# Update last metrics to avoid immediate duplicate
					if 'center_now' in locals():
						last_center = center_now  # type: ignore[assignment]
					if 'area_now' in locals():
						last_area = area_now  # type: ignore[assignment]
					last_accept_time = time.time()
					print(f"[calibrate] Captured sample {len(imgpoints)}/{samples_needed} (manual)")
					status_flash_msg = "Manual capture ✓"
					status_flash_until = time.time() + 0.8
				else:
					# Feedback when pattern not detected
					status_flash_msg = "Pattern not found – adjust distance/angle/focus"
					status_flash_until = time.time() + 1.2
			if key == ord("p"):
				# Toggle picture-in-picture preview
				show_preview = not show_preview
			if key == ord("i"):
				# Invert the on-screen chessboard to help detection under different exposures
				invert_pattern = not invert_pattern
				pattern = _gen_chessboard(board_cols, board_rows, square_px=square_px, margin_px=local_margin_px, invert=invert_pattern)

			if len(imgpoints) >= samples_needed:
				break
	finally:
		conn.close()
		try:
			cv2.destroyWindow(target_window)
		except Exception:
			pass
		# Extra teardown to ensure windows and UI close cleanly across calibrations
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass
		try:
			for _ in range(5):
				cv2.waitKey(1)
		except Exception:
			pass
		try:
			filesystem.stop_rdmp_stream()
		except Exception:
			pass

	if not imgpoints:
		raise RuntimeError("No calibration samples were captured. Ensure the pattern is visible to the camera.")

	# Run calibration
	h, w = gray.shape[:2]  # type: ignore[name-defined]
	ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

	# Save to device
	try:
		filesystem.create_directory("/calibration")
	except Exception:
		pass

	filename = "intrinsics.json"
	if label:
		filename = f"intrinsics_{label}.json"
	device_path = f"/calibration/{filename}"

	payload = _format_intrinsics_json(label or "camera", (w, h), mtx, dist, ret, len(imgpoints))
	resp = filesystem.write_file(device_path, payload)
	ok = isinstance(resp, dict) and resp.get("status") == "success"
	# Verify file exists; if not, retry with append chunks (defensive against transport limits)
	exists = False
	try:
		exists = bool(filesystem.file_exists(device_path))
	except Exception:
		exists = False
	if not exists or not ok:
		try:
			# Clean up any partial file
			try:
				filesystem.delete_file(device_path)
			except Exception:
				pass
			# Append in small chunks
			chunk_size = 512
			for i in range(0, len(payload), chunk_size):
				chunk = payload[i : i + chunk_size]
				filesystem.append_file(device_path, chunk)
			# Re-verify
			exists = bool(filesystem.file_exists(device_path))
		except Exception:
			exists = False
	if not exists:
		raise RuntimeError("Failed to persist intrinsics file on device")

	return CalibrationResult(
		label=label or "camera",
		image_size=(w, h),
		camera_matrix=mtx.tolist(),
		dist_coeffs=dist.reshape(-1).tolist(),
		rms=float(ret),
		num_samples=len(imgpoints),
		device_path=device_path,
		timestamp=_utc_now_iso(),
	)


