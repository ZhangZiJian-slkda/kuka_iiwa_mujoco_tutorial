"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2025-12-31 15:10:19
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-12-31 15:10:19
FilePath: /kuka_iiwa_mujoco_tutorial/Trajectory/SensorCalibration.py
Copyright (c) 2025 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
import mujoco
import mujoco.viewer
import numpy as np
import time


def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""
    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints

    def _generate_trajectory(self):
        """生成插值轨迹"""
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        yield self.end_joints  # 确保最后精确到达目标值

    def get_next_waypoint(self, qpos):
        """返回下一个轨迹点"""
        if np.allclose(qpos, self.waypoint, atol=0.02):
            try:
                self.waypoint = next(self.trajectory)
                return self.waypoint
            except StopIteration:
                pass
        return self.waypoint


def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    # ✅ 给定的角度制数据（单位：degree）
    doublelllljoints_deg = [
        [0.0, 16.18, 23.04, 37.35, -67.93, 38.14, -2.13],
        [18.51, 9.08, -1.90, 49.58, -2.92, 18.60, -31.18],
        [-18.53, -25.76, -47.03, -49.55, 30.76, -30.73, 20.11],
        [-48.66, 24.68, -11.52, 10.48, -11.38, -20.70, 20.87],
        [9.01, -35.00, 24.72, -82.04, 14.65, -29.95, 1.57]
    ]

    # ✅ 转换为弧度制（rad）
    joint_angle_list = np.deg2rad(doublelllljoints_deg)

    # 设置起始姿态
    start_joints = joint_angle_list[0]
    data.ctrl[:7] = start_joints
    data.qpos[:7] = start_joints

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        print("开始执行五组角度轨迹（由degree→radian转换）...")

        for i in range(1, len(joint_angle_list)):
            end_joints = joint_angle_list[i]
            print(f"正在执行第 {i} 组目标姿态（弧度制）：{np.round(end_joints, 3)}")

            # 生成线性插值轨迹
            joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=150)

            while viewer.is_running():
                waypoint = joint_trajectory.get_next_waypoint(data.qpos[:7])
                data.ctrl[:7] = waypoint
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

                # 判断是否完成当前阶段
                if np.allclose(data.qpos[:7], end_joints, atol=0.02):
                    print(f"第 {i} 组姿态完成 ✅")
                    break

            # 当前末端状态作为下一段的起点
            start_joints = np.copy(end_joints)
            time.sleep(1.0)

        print("全部五组关节角执行完毕 ✅")

        # 保持窗口，供观察
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
