## Voxel SDK

Concise tools to control a Voxel device over wired serial or BLE, capture media, and stream to your computer.

### Install

Default install includes everything (BLE, Serial, and the stream viewer):

```bash
pip install voxel-sdk
```

Optional (for video conversion): install `ffmpeg` (e.g., `brew install ffmpeg` on macOS).


## Quickstart

### 1) Terminal (interactive CLI)

Run the built-in terminal from the repository:

```bash
cd voxel-sdk
python terminal.py
```

- Choose connection:
  - Wired: select “Wired” or run with flags: `python terminal.py --transport serial --port /dev/cu.usbmodem1101`
  - BLE: select “Bluetooth” or run with flags: `python terminal.py --transport ble --ble-name voxel`

Once connected you’ll see a prompt like `voxel>`. A few useful commands:

- List files:

```text
voxel> ls /
```

- Capture a photo to the device:

```text
voxel> camera-capture /photos myphoto 1600x1200
voxel> ls /photos
```

- Download the photo to your computer (current directory):

```text
voxel> download /photos/myphoto.jpg myphoto.jpg
```

- Stream live video to your computer with a local viewer (press `q` to quit):

```text
voxel> stream 9000
```

Notes:
- Stream viewer requires OpenCV + NumPy (installed by default). If you built a minimal env without them, install `opencv-python` and `numpy`.
- You can stop a remote stream with `stream-stop`.
- The CLI isn’t currently exposed as a package module. Use `python terminal.py` from the repo (or create your own entrypoint in your app using the SDK).


### 2) Python SDK

Use the high-level controller with Serial or BLE transports.

- Take a photo and save it locally:

```python
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.ble import BleVoxelTransport  # or: from voxel_sdk.serial import SerialVoxelTransport

# Connect (BLE)
transport = BleVoxelTransport(device_name="voxel")
transport.connect("")  # scans for a device whose name starts with "voxel"
controller = DeviceController(transport)

# Ask the device to capture a photo to its filesystem
controller.execute_device_command("camera_capture:/photos|myphoto|640x480")

# Download the photo to your computer
jpeg_bytes = controller.download_file("/photos/myphoto.jpg")
with open("myphoto.jpg", "wb") as f:
    f.write(jpeg_bytes)

transport.disconnect()
```

- Stream live video with a local viewer:

```python
from voxel_sdk.device_controller import DeviceController
from voxel_sdk.ble import BleVoxelTransport  # or SerialVoxelTransport

transport = BleVoxelTransport(device_name="voxel")
transport.connect("")
controller = DeviceController(transport)

# Opens an OpenCV window; press 'q' to quit
controller.stream_with_visualization(port=9000)

transport.disconnect()
```

Tips:
- On macOS, serial ports often look like `/dev/cu.usbmodem*` or `/dev/tty.usbserial*`.
- If you’re unsure of the saved image filename, run `ls /photos` in the terminal first, then use that path in `download`/`download_file`.


## Extras

- Default install already includes these. Extras exist only if you want a minimal/custom install:
  - `ble`: Bluetooth Low Energy (via `bleak`)
  - `serial`: Wired serial (via `pyserial`)
  - `viz`: Local viewer for streaming (via `opencv-python`, `numpy`)
  - `all`: Installs all of the above


## Links

- Source: https://github.com/physicalinc/voxel-sdk


