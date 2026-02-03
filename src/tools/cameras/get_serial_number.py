import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
print("Connected RealSense devices:")

for i, dev in enumerate(devices):
    print(f"Device {i}: {dev.get_info(rs.camera_info.serial_number)}")
