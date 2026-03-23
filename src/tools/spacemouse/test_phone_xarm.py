import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
from asmagic import ARDataSubscriber
from xarm.wrapper import XArmAPI


XARM_IP = "192.168.1.230"
PHONE_IP = "192.168.1.95"
TRANSLATION_SCALE = 1000.0  # meters -> mm
LOOP_HZ = 50.0
MAX_TCP_SPEED = 90.0        # mm/s – reduced-mode TCP speed cap
LOOP_DT = 1.0 / LOOP_HZ
MAX_ROT_DEG = 60.0          # maximum rotation delta from init (degrees)
    
# Phone frame (asmagic robotics): X=Forward, Y=Left, Z=Up
# xArm frame (right-handed):      X=Forward,  Y=Left, Z=Up
# Mapping: phone_-X→robot_X, phone_Y→robot_Y, phone_Z→robot_Z
# det=-1 (single-axis reflection), stored as plain numpy matrix (not scipy Rotation)
M_PHONE_TO_ROBOT = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=float)


class PhoneSubscriber:
    """Background thread that continuously reads local_pose from ARDataSubscriber."""

    def __init__(self):
        self._sub = ARDataSubscriber(PHONE_IP)
        self._lock = threading.Lock()
        self._latest_pose = None  # [tx, ty, tz, qx, qy, qz, qw]
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            for data in self._sub:
                with self._lock:
                    self._latest_pose = np.array(data.local_pose, dtype=np.float64)
        except Exception as e:
            print(f"PhoneSubscriber error: {e}")

    def get_pose(self):
        """Return latest [tx, ty, tz, qx, qy, qz, qw] or None."""
        with self._lock:
            return self._latest_pose.copy() if self._latest_pose is not None else None

    def close(self):
        self._sub.close()


def phone_to_robot_pose(
    phone_pose: np.ndarray,
    init_phone: np.ndarray,
    init_arm: np.ndarray,
) -> np.ndarray:
    """Compute target arm pose [x,y,z,roll,pitch,yaw] (mm, rad).

    init_phone is captured at subscriber startup — local_pose is relative to
    when the AR session started, NOT when the subscriber connects, so the
    offset must be removed explicitly.
    """
    # --- Position ---
    # M has det=-1 (LH→RH), so use plain matmul instead of Rotation.apply
    dpos_robot = M_PHONE_TO_ROBOT @ ((phone_pose[:3] - init_phone[:3]) * TRANSLATION_SCALE)
    target_pos = init_arm[:3] + dpos_robot

    # --- Orientation ---
    # World-frame delta: R_delta s.t. R_delta * R_init = R_current
    # (right-multiply form r_init_inv gives body-frame delta, which is wrong here;
    #  we need world-frame delta so that M @ R_delta @ M^T transforms correctly)
    r_init_phone = Rotation.from_quat(init_phone[3:7])
    r_current = Rotation.from_quat(phone_pose[3:7])
    drot_phone = r_current * r_init_phone.inv()

    # Clamp rotation magnitude to MAX_ROT_DEG to prevent workspace violations
    rotvec = drot_phone.as_rotvec()
    angle = np.linalg.norm(rotvec)
    max_rad = np.deg2rad(MAX_ROT_DEG)
    if angle > max_rad:
        rotvec = rotvec / angle * max_rad
        drot_phone = Rotation.from_rotvec(rotvec)

    drot_robot = Rotation.from_matrix(M_PHONE_TO_ROBOT @ drot_phone.as_matrix() @ M_PHONE_TO_ROBOT.T)

    # Compose with initial arm orientation (world-frame: dR applied in base frame)
    # xArm RPY = extrinsic roll-pitch-yaw → scipy lowercase "xyz"
    r_init_arm = Rotation.from_euler("xyz", init_arm[3:6])
    target_rpy = (drot_robot * r_init_arm).as_euler("xyz")

    return np.concatenate([target_pos, target_rpy])


def setup_arm() -> XArmAPI:
    arm = XArmAPI(XARM_IP, is_radian=True)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_reduced_max_tcp_speed(MAX_TCP_SPEED)
    arm.set_reduced_mode(True)
    arm.set_mode(1)  # servo motion mode
    arm.set_state(0)
    return arm


def main():
    print(f"Connecting to xArm at {XARM_IP}")
    arm = setup_arm()

    code, init_arm_pose = arm.get_position()
    if code != 0 or init_arm_pose is None:
        raise RuntimeError(f"Failed to read initial arm pose, code={code}")
    init_arm_pose = np.array(init_arm_pose, dtype=np.float64)
    print(f"Initial arm pose: {init_arm_pose}")

    print(f"Connecting to phone at {PHONE_IP} ...")
    phone = PhoneSubscriber()
    init_phone_pose = None
    while init_phone_pose is None:
        init_phone_pose = phone.get_pose()
        time.sleep(0.05)
    print(f"Phone connected. Init phone pose: {np.round(init_phone_pose, 4)}")

    print("Phone control loop started (servo mode). Press Ctrl+C to exit.")
    print("  Phone pose columns: [tx, ty, tz,  qx, qy, qz, qw]  (meters)")
    print("  Robot target columns: [x, y, z,  roll, pitch, yaw]  (mm, rad)")
    _diag_count = 0
    try:
        while True:
            phone_pose = phone.get_pose()
            if phone_pose is None:
                time.sleep(LOOP_DT)
                continue

            cur_pose = phone_to_robot_pose(phone_pose, init_phone_pose, init_arm_pose)

            # Print diagnostics at ~2 Hz so axes/signs are easy to verify
            _diag_count += 1
            if _diag_count % (int(LOOP_HZ) // 2) == 0:
                pp = np.round(phone_pose, 4)
                cp = np.round(cur_pose, 2)
                print(f"phone [{pp[0]:7.4f} {pp[1]:7.4f} {pp[2]:7.4f} | "
                      f"{pp[3]:6.3f} {pp[4]:6.3f} {pp[5]:6.3f} {pp[6]:6.3f}]  "
                      f"→ robot [{cp[0]:7.1f} {cp[1]:7.1f} {cp[2]:7.1f} | "
                      f"{cp[3]:6.3f} {cp[4]:6.3f} {cp[5]:6.3f}]")

            code = arm.set_servo_cartesian(cur_pose.tolist())
            if code != 0:
                print(f"set_servo_cartesian failed with code {code}")

            time.sleep(LOOP_DT)
    except KeyboardInterrupt:
        print("\nStopping control loop...")
    finally:
        phone.close()
        arm.disconnect()


if __name__ == "__main__":
    main()
