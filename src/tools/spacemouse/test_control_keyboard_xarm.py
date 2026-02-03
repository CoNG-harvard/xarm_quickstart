"""
Test keyboard control of xArm.

Usage:

- servo mode: 
    python ./src/tools/spacemouse/test_control_keyboard_xarm.py --mode servo
    
- position mode:
    python ./src/tools/spacemouse/test_control_keyboard_xarm.py --mode position

"""


import os
import time
from typing import Dict, Optional
from collections import deque

import numpy as np
from pynput.keyboard import Listener  # type: ignore
from xarm.wrapper import XArmAPI


import argparse

DEFAULT_IP = os.environ.get("XARM_IP", "192.168.1.230")
SLEEP_SEC = 0.1
KEY_TRANSLATION_STEP_MM = 3.0
KEY_ROTATION_STEP_DEG = 3.0
DEFAULT_SPEED = 50  # mm/s or deg/s
DEFAULT_ACC = 1000  # mm/s^2 or deg/s^2
LOOP_HZ = 50.0
LOOP_DT = 1.0 / LOOP_HZ


def setup_arm(ip: str, mode: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=False)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0 if mode == "position" else 1)
    arm.set_state(0)
    return arm


class KeyboardOnceController:
    def __init__(self):
        self.queue: deque[Dict[str, np.ndarray]] = deque(maxlen=1)
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()

    def _display_controls(self):
        print("\nKeyboard Controls Active (single press moves once):")
        print("Position (mm): w/s = +Y/-Y, a/d = -X/+X, q/e = +Z/-Z")
        print("Rotation (deg): u/j = +Roll/-Roll, i/k = +Pitch/-Pitch, o/l = +Yaw/-Yaw")
        print("Press Ctrl+C to exit.\n")

    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return True

        dpos = np.zeros(3)
        drot = np.zeros(3)

        if k == "w":
            dpos[1] += KEY_TRANSLATION_STEP_MM
        elif k == "s":
            dpos[1] -= KEY_TRANSLATION_STEP_MM
        elif k == "a":
            dpos[0] -= KEY_TRANSLATION_STEP_MM
        elif k == "d":
            dpos[0] += KEY_TRANSLATION_STEP_MM
        elif k == "q":
            dpos[2] += KEY_TRANSLATION_STEP_MM
        elif k == "e":
            dpos[2] -= KEY_TRANSLATION_STEP_MM
        elif k == "u":
            drot[0] += KEY_ROTATION_STEP_DEG
        elif k == "j":
            drot[0] -= KEY_ROTATION_STEP_DEG
        elif k == "i":
            drot[1] += KEY_ROTATION_STEP_DEG
        elif k == "k":
            drot[1] -= KEY_ROTATION_STEP_DEG
        elif k == "o":
            drot[2] += KEY_ROTATION_STEP_DEG
        elif k == "l":
            drot[2] -= KEY_ROTATION_STEP_DEG
        else:
            return True

        self.queue.append({"dpos": dpos, "drot": drot})
        return True

    def on_release(self, key):
        return True

    def next_command(self) -> Optional[Dict[str, np.ndarray]]:
        if self.queue:
            return self.queue.popleft()
        return None


def main(args: argparse.Namespace):
    ip = DEFAULT_IP
    print(f"Connecting to xArm at {ip}")
    arm = setup_arm(ip, args.mode)
    print("Connected. Enabling keyboard control...")
    controller = KeyboardOnceController()

    try:
        while True:
            cmd = controller.next_command()
            if cmd is None:
                dpos = np.zeros(3)
                drot = np.zeros(3)
            else:
                dpos = cmd["dpos"]
                drot = cmd["drot"]

            # Always output current command (zeros when idle)
            print(f"cmd dpos={dpos} drot={drot}")

            if np.any(dpos) or np.any(drot):
                if args.mode == "position":
                    code = arm.set_position(
                        x=dpos[0],
                        y=dpos[1],
                        z=dpos[2],
                        roll=drot[0],
                        pitch=drot[1],
                        yaw=drot[2],
                        relative=True,
                        is_radian=False,
                        wait=True,
                    )
                else:  # servo
                    code, cur_pos = arm.get_position(is_radian=False)
                    if code != 0:
                        print(f"get_position failed with code {code}")
                        time.sleep(LOOP_DT)
                        continue
                    code = arm.set_servo_cartesian(
                        mvpose=[
                            cur_pos[0] + dpos[0],
                            cur_pos[1] + dpos[1],
                            cur_pos[2] + dpos[2],
                            cur_pos[3] + drot[0],
                            cur_pos[4] + drot[1],
                            cur_pos[5] + drot[2],
                        ],
                        is_radian=False,
                    )
                if code != 0:
                    print(f"set_position failed with code {code}")

            time.sleep(LOOP_DT)
    except KeyboardInterrupt:
        print("Stopping control loop...")
    finally:
        arm.disconnect()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test keyboard control of xArm.")
    parser.add_argument("--mode", type=str, default="pos", choices=["position", "servo"],
                        help="Mode to run the control in. Options: position, servo.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
