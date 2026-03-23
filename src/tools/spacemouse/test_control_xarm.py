import os
import threading
import time
from typing import Dict

import numpy as np
import pyspacemouse
from pynput.keyboard import Listener  # type: ignore
from xarm.wrapper import XArmAPI


DEFAULT_IP = os.environ.get("XARM_IP", "192.168.1.230")
TRANSLATION_SCALE_MM = 1.5
TRANSLATION_DEADZONE_MM = 0.4
ROTATION_SCALE_DEG = 1.0
ROTATION_DEADZONE_DEG = 0.4
LOOP_HZ = 50.0
LOOP_DT = 1.0 / LOOP_HZ


class SpaceMouse:
    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0):
        self._control = [0.0] * 6
        self._toggle_gripper = False
        self._prev_left = False
        self._reset_state = 0
        self._enabled = False
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.device = pyspacemouse.open(
            dof_callback=self._dof_callback, button_callback=self._button_callback
        )
        if not self.device:
            raise RuntimeError("Failed to connect to SpaceMouse. Ensure spacenavd is stopped.")

        self._display_controls()

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _dof_callback(self, state):
        if self._enabled:
            self._control = [
                state.y,
                state.x,
                state.z * -1.0,
                state.roll,
                state.pitch,
                state.yaw,
            ]

    def _button_callback(self, state, buttons):
        # Left button toggles gripper on press
        left_pressed = bool(buttons[0])
        if left_pressed and not self._prev_left:
            self._toggle_gripper = True
        self._prev_left = left_pressed

        if buttons[1]:
            self._reset_state = 1
            self._enabled = False

    def run(self):
        while True:
            if self.device:
                self.device.read()
            time.sleep(0.01)

    def get_controller_state(self) -> Dict[str, np.ndarray]:
        dpos = self.control[:3] * TRANSLATION_SCALE_MM * self.pos_sensitivity
        drot_deg = self.control[3:] * ROTATION_SCALE_DEG * self.rot_sensitivity
        # apply deadzones
        dpos = np.where(np.abs(dpos) < TRANSLATION_DEADZONE_MM, 0.0, dpos)
        drot_deg = np.where(np.abs(drot_deg) < ROTATION_DEADZONE_DEG, 0.0, drot_deg)
        return dict(dpos=dpos, drot_deg=drot_deg, grasp=self.control_gripper, reset=self._reset_state)

    @property
    def control(self):
        return np.array(self._control)

    @property
    def control_gripper(self):
        return 1.0 if self._toggle_gripper else 0.0

    def clear_gripper_toggle(self):
        self._toggle_gripper = False

    def start_control(self):
        self._reset_state = 0
        self._enabled = True

    def _display_controls(self):
        print("\nSpaceMouse Controls Active:")
        print("Puck Movement: Move Arm (relative)")
        print("Left Button:  Close Gripper (Hold)")
        print("Right Button: Reset Arm\n")

    # Placeholder to keep listener active; close on Ctrl+C
    def on_press(self, key):
        return True

    def on_release(self, key):
        return True


def setup_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, is_radian=False)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(1)  # servo motion mode
    arm.set_state(0)
    return arm


def run_spacemouse_profile(ip: str):
    print(f"Connecting to xArm at {ip}")
    arm = setup_arm(ip)

    mouse = SpaceMouse()
    mouse.start_control()

    # track current pose for incremental servo commands
    code, cur_pose = arm.get_position(is_radian=False)
    if code != 0 or cur_pose is None:
        raise RuntimeError(f"Failed to read initial pose, code={code}")

    gripper_closed = False
    print("SpaceMouse control loop started (servo mode). Press Ctrl+C to exit.")

    try:
        while True:
            state = mouse.get_controller_state()
            dpos = state["dpos"]
            drot = state["drot_deg"]

            # Always output current command (zero when no input)
            # print(f"cmd dpos={dpos} drot={drot}")

            if np.any(dpos) or np.any(drot):
                # incremental update of cached pose
                cur_pose = [
                    cur_pose[0] + dpos[0],
                    cur_pose[1] + dpos[1],
                    cur_pose[2] + dpos[2],
                    cur_pose[3] + drot[0],
                    cur_pose[4] + drot[1],
                    cur_pose[5] + drot[2],
                ]
                code = arm.set_servo_cartesian(cur_pose, is_radian=False)
                if code != 0:
                    print(f"set_servo_cartesian failed with code {code}")

            grasp_now = bool(state["grasp"])
            if grasp_now:
                target = 850 if not gripper_closed else 100
                arm.set_gripper_position(target, wait=False, speed=1000, auto_enable=True)
                gripper_closed = not gripper_closed
                mouse.clear_gripper_toggle()

            if state["reset"]:
                mouse._reset_state = 0
                print("Aligning gripper to [180, 0, 0]...")
                # keep current position, set orientation
                code, cur_pose = arm.get_position(is_radian=False)
                if code == 0 and cur_pose is not None:
                    cur_pose = [
                        cur_pose[0],
                        cur_pose[1],
                        cur_pose[2],
                        180.0,
                        0.0,
                        0.0,
                    ]
                    code = arm.set_servo_cartesian(cur_pose, is_radian=False)
                if code != 0:
                    print(f"align failed with code {code}")
                mouse.start_control()

            time.sleep(LOOP_DT)
    except KeyboardInterrupt:
        print("Stopping control loop...")
    finally:
        try:
            pyspacemouse.close()
        except Exception:
            pass
        arm.disconnect()


def main():
    ip = DEFAULT_IP
    run_spacemouse_profile(ip)


if __name__ == "__main__":
    main()