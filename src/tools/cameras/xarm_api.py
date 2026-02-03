# A sample script to control the xarm robot

from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.230')
arm.motion_enable(enable=True)
arm.set_mode(0)  # set to position control mode
arm.set_state(0)  # set to ready state

# get the current end effector position
current_pos = arm.get_position()[1]
x, y, z, r, p, y = current_pos[0], current_pos[1], current_pos[2], current_pos[3], current_pos[4], current_pos[5]
print(f"End effector position: {x}, {y}, {z}, {r}, {p}, {y}")

# get the joint angles
joint_states = arm.get_joint_states()[1]
joint_angles = joint_states[0]
j1, j2, j3, j4, j5, j6, j7 = joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], joint_angles[5], joint_angles[6]
print(f"Joint angles: {j1}, {j2}, {j3}, {j4}, {j5}, {j6}, {j7}")

# get the gripper position, max is ~840, min is 0.
gripper_state = arm.get_gripper_position()
print(f"Gripper state: {gripper_state[0]}, Gripper position: {gripper_state[1]}")

arm.disconnect()