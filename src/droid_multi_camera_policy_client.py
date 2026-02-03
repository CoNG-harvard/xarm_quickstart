"""Stream two RealSense cameras to the policy server (DROID env).

- Wrist camera (serial 323522063283 by default) is mapped to
  `observation/wrist_image_left`.
- The second camera is mapped to `observation/exterior_image_1_left`.

Frames are captured on a background thread with a single-slot queue so the
client always sends the freshest frame and does not build latency. Resize with
padding keeps aspect ratio and matches the default 224x224 input expected by
the policy server. Robot state is pulled asynchronously from XArmAPI to avoid
blocking inference; joint and gripper values are still easy to override if you
want to provide your own sources.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# Ensure we can import the local openpi_client package when running from repo root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OPENPI_SRC = PROJECT_ROOT / "openpi-client" / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

from openpi_client import image_tools  # type: ignore  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: Optional[int] = 8000
    api_key: Optional[str] = None
    wrist_serial: str = "323522063283"
    exterior_serial: str = "234422060060"
    width: int = 640
    height: int = 480
    target_image_size: int = 224
    prompt: str = "do something"
    log_every: float = 2.0  # seconds
    arm_ip: str = "192.168.1.230"
    arm_poll_hz: float = 20.0
    arm_enabled: bool = True
    visualize: bool = True
    viz_scale: float = 1.5


class DualRealSenseStreamer:
    """Capture wrist + exterior streams on a background thread."""

    def __init__(self, wrist_serial: str, exterior_serial: str, width: int, height: int):
        self._wrist_serial = wrist_serial
        self._exterior_serial = exterior_serial
        self._width = width
        self._height = height
        self._queue: queue.Queue[Tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._pipeline_wrist = rs.pipeline()
        self._pipeline_exterior = rs.pipeline()

    def start(self) -> None:
        config_wrist = rs.config()
        config_wrist.enable_device(self._wrist_serial)
        config_wrist.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, 30)

        config_exterior = rs.config()
        config_exterior.enable_device(self._exterior_serial)
        config_exterior.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, 30)

        self._pipeline_wrist.start(config_wrist)
        self._pipeline_exterior.start(config_exterior)

        self._thread.start()
        logger.info("DualRealSenseStreamer started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._pipeline_wrist.stop()
        self._pipeline_exterior.stop()
        logger.info("DualRealSenseStreamer stopped.")

    def read_latest(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames_wrist = self._pipeline_wrist.wait_for_frames()
                frames_exterior = self._pipeline_exterior.wait_for_frames()

                wrist_frame = frames_wrist.get_color_frame()
                exterior_frame = frames_exterior.get_color_frame()
                if not wrist_frame or not exterior_frame:
                    continue

                wrist_img = cv2.cvtColor(np.asanyarray(wrist_frame.get_data()), cv2.COLOR_BGR2RGB)
                exterior_img = cv2.cvtColor(np.asanyarray(exterior_frame.get_data()), cv2.COLOR_BGR2RGB)

                # Keep only the freshest frame to avoid latency accumulation.
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put_nowait((wrist_img, exterior_img))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Capture loop error: %s", exc, exc_info=True)
                time.sleep(0.1)


class XArmStatePoller:
    """Continuously pull joint + gripper state from an xArm without blocking inference."""

    def __init__(self, arm_ip: str, poll_hz: float) -> None:
        self._arm_ip = arm_ip
        self._poll_interval = 1.0 / max(poll_hz, 1.0)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._lock = threading.Lock()
        self._latest_joint: Optional[np.ndarray] = None
        self._latest_gripper: Optional[np.ndarray] = None
        self._arm: Optional[XArmAPI] = None

    def start(self) -> None:
        try:
            self._arm = XArmAPI(self._arm_ip)
            self._arm.motion_enable(enable=True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            self._thread.start()
            logger.info("Connected to xArm at %s", self._arm_ip)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to init xArmAPI: %s", exc, exc_info=True)
            self._arm = None

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._arm:
            try:
                self._arm.disconnect()
            except Exception:
                pass
        logger.info("xArmStatePoller stopped.")

    def read_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._latest_joint, self._latest_gripper

    def _poll_loop(self) -> None:
        if self._arm is None:
            return
        while not self._stop_event.is_set():
            try:
                ret_joint = self._arm.get_joint_states()
                if ret_joint and ret_joint[0] == 0:
                    joint_angles = np.array(ret_joint[1][0], dtype=np.float32)
                else:
                    joint_angles = None

                ret_gripper = self._arm.get_gripper_position()
                if ret_gripper and ret_gripper[0] == 0:
                    gripper_pos = np.array([ret_gripper[1]], dtype=np.float32)
                else:
                    gripper_pos = None

                with self._lock:
                    if joint_angles is not None:
                        self._latest_joint = joint_angles
                    if gripper_pos is not None:
                        self._latest_gripper = gripper_pos
            except Exception as exc:  # noqa: BLE001
                logger.warning("xArm poll error: %s", exc, exc_info=True)
            time.sleep(self._poll_interval)


def _prepare_image(img: np.ndarray, target: int) -> np.ndarray:
    """Resize with padding to target x target and ensure uint8."""
    img = image_tools.resize_with_pad(img, target, target)
    img = image_tools.convert_to_uint8(img)
    return img


def build_observation(
    wrist_img: np.ndarray,
    exterior_img: np.ndarray,
    target_size: int,
    prompt: str,
    joint_position: np.ndarray,
    gripper_position: np.ndarray,
) -> dict:
    return {
        "observation/wrist_image_left": _prepare_image(wrist_img, target_size),
        "observation/exterior_image_1_left": _prepare_image(exterior_img, target_size),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }


def _draw_status(
    img: np.ndarray,
    latency_ms: float,
    joint: np.ndarray,
    gripper: np.ndarray,
    prompt: str,
) -> np.ndarray:
    """Overlay latency + state text on an RGB image; returns BGR for imshow."""
    disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    y = 18
    for line in [
        f"Latency: {latency_ms:.1f} ms",
        f"Joints: {np.array2string(joint, precision=2, suppress_small=True)}",
        f"Gripper: {np.array2string(gripper, precision=2, suppress_small=True)}",
        f"Prompt: {prompt}",
    ]:
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return disp


def _visualize_observation(
    obs: dict,
    latency_ms: float,
    prompt: str,
    scale: float,
) -> None:
    wrist = obs["observation/wrist_image_left"]
    exterior = obs["observation/exterior_image_1_left"]
    joint = obs["observation/joint_position"]
    gripper = obs["observation/gripper_position"]

    wrist_disp = _draw_status(wrist, latency_ms, joint, gripper, prompt)
    exterior_disp = cv2.cvtColor(exterior, cv2.COLOR_RGB2BGR)

    h, w, _ = wrist_disp.shape
    if exterior_disp.shape[:2] != (h, w):
        exterior_disp = cv2.resize(exterior_disp, (w, h))

    combined = np.hstack([wrist_disp, exterior_disp])
    if scale != 1.0:
        combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("DROID Observation (wrist | exterior)", combined)
    cv2.waitKey(1)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Stream RealSense images to DROID policy server.")
    parser.add_argument("--host", type=str, default=Args.host, help="Policy server host or ws:// URI.")
    parser.add_argument("--port", type=int, default=Args.port, help="Policy server port.")
    parser.add_argument("--api-key", type=str, default=Args.api_key, help="Optional API key.")
    parser.add_argument("--wrist-serial", type=str, default=Args.wrist_serial, help="Wrist camera serial.")
    parser.add_argument("--exterior-serial", type=str, default=Args.exterior_serial, help="Exterior camera serial.")
    parser.add_argument("--width", type=int, default=Args.width, help="Capture width.")
    parser.add_argument("--height", type=int, default=Args.height, help="Capture height.")
    parser.add_argument("--target-image-size", type=int, default=Args.target_image_size, help="Model input size.")
    parser.add_argument("--prompt", type=str, default=Args.prompt, help="Task prompt.")
    parser.add_argument("--log-every", type=float, default=Args.log_every, help="Seconds between latency logs.")
    parser.add_argument("--arm-ip", type=str, default=Args.arm_ip, help="xArm robot IP.")
    parser.add_argument("--arm-poll-hz", type=float, default=Args.arm_poll_hz, help="xArm state polling rate.")
    parser.add_argument("--no-arm", action="store_true", help="Disable xArm polling and send zero state.")
    parser.add_argument("--no-viz", action="store_true", help="Disable realtime observation visualization.")
    parser.add_argument("--viz-scale", type=float, default=Args.viz_scale, help="Scale factor for cv2 window.")
    parsed = parser.parse_args()
    return Args(
        host=parsed.host,
        port=parsed.port,
        api_key=parsed.api_key,
        wrist_serial=parsed.wrist_serial,
        exterior_serial=parsed.exterior_serial,
        width=parsed.width,
        height=parsed.height,
        target_image_size=parsed.target_image_size,
        prompt=parsed.prompt,
        log_every=parsed.log_every,
        arm_ip=parsed.arm_ip,
        arm_poll_hz=parsed.arm_poll_hz,
        arm_enabled=not parsed.no_arm,
        visualize=not parsed.no_viz,
        viz_scale=parsed.viz_scale,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    streamer = DualRealSenseStreamer(
        wrist_serial=args.wrist_serial,
        exterior_serial=args.exterior_serial,
        width=args.width,
        height=args.height,
    )
    streamer.start()

    poller: Optional[XArmStatePoller] = None
    if args.arm_enabled:
        poller = XArmStatePoller(arm_ip=args.arm_ip, poll_hz=args.arm_poll_hz)
        poller.start()

    policy = WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    logger.info("Connected to policy server with metadata: %s", policy.get_server_metadata())

    last_log = time.time()
    try:
        while True:
            latest = streamer.read_latest(timeout=1.0)
            if latest is None:
                logger.warning("No frames received yet.")
                continue

            wrist_img, exterior_img = latest
            joint_state = np.zeros((7,), dtype=np.float32)
            gripper_state = np.zeros((1,), dtype=np.float32)
            if poller is not None:
                latest_joint, latest_gripper = poller.read_latest()
                if latest_joint is not None:
                    joint_state = latest_joint.astype(np.float32)
                if latest_gripper is not None:
                    gripper_state = latest_gripper.astype(np.float32)

            obs = build_observation(
                wrist_img=wrist_img,
                exterior_img=exterior_img,
                target_size=args.target_image_size,
                prompt=args.prompt,
                joint_position=joint_state,
                gripper_position=gripper_state,
            )

            start = time.time()
            action = policy.infer(obs)
            latency_ms = 1000 * (time.time() - start)

            now = time.time()
            if now - last_log >= args.log_every:
                logger.info("Inference latency: %.1f ms | action keys: %s", latency_ms, list(action.keys()))
                last_log = now

            if args.visualize:
                _visualize_observation(obs, latency_ms=latency_ms, prompt=args.prompt, scale=args.viz_scale)
    except KeyboardInterrupt:
        logger.info("Stopping client...")
    finally:
        streamer.stop()
        if poller is not None:
            poller.stop()
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
