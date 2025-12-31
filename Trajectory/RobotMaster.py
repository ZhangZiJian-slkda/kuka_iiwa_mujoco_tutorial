'''
Author: Zhang-sklda
Date: 2025-12-31
LastEditors: ChatGPT (slow jog version)
FilePath: /kuka_iiwa_mujoco_tutorial/master_calibration_slow.py
Description: 模拟机械臂掉轴后单关节jog操作（带速度控制与暂停）
'''
import mujoco
import mujoco.viewer
import numpy as np
import time


def viewer_init(viewer):
    """初始化MuJoCo视角"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class JointSpaceJogController:
    """单关节Jog控制器（低速版本）"""
    def __init__(self, model, data, joint_index, target_angle_rad, steps=400):
        self.model = model
        self.data = data
        self.joint_index = joint_index
        self.target = target_angle_rad
        self.steps = steps
        self.start = data.qpos[joint_index]
        self.step = (self.target - self.start) / self.steps
        self.current = self.start
        self.counter = 0

    def step_once(self):
        """执行单步Jog"""
        if abs(self.current - self.target) > 1e-3 and self.counter < self.steps:
            self.current += self.step
            self.data.qpos[self.joint_index] = self.current
            self.data.ctrl[self.joint_index] = self.current
            mujoco.mj_step(self.model, self.data)
            self.counter += 1
            return True
        else:
            return False


def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    ee_site_id = model.site("attachment_site").id  # 末端执行器 site

    # -------------------
    # ✅ 初始状态设置（4轴= -1.57 rad）
    # -------------------
    init_joints = np.array([0.15, 0.3, 0.10, -1.57, 0.2, 0.2, 0.09])
    data.qpos[:7] = init_joints
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    data.ctrl[:7] = np.copy(data.qpos[:7])
    print("✅ 初始姿态设置成功：4轴 = -1.57 rad (≈ -90°)")

    # -------------------
    # Jog操作顺序：6→2→4轴
    # -------------------
    jog_plan_deg = [
        (5, 90),   # 6轴 → 90°
        (1, 50),   # 2轴 → 50°
        (3, 0)     # 4轴 → 0°
    ]
    jog_plan_rad = [(idx, np.deg2rad(angle)) for idx, angle in jog_plan_deg]

    SAFE_HEIGHT = 1.2  # 末端执行器Z高度上限（米）

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        print("开始机械臂 master 标定过程（单关节 jog 模式，低速+暂停）...")

        for i, (joint_id, target_angle_rad) in enumerate(jog_plan_rad):
            print(f"\n👉 [{i+1}/3] 开始移动第 {joint_id+1} 轴至 {np.rad2deg(target_angle_rad):.1f}°")
            controller = JointSpaceJogController(model, data, joint_id, target_angle_rad, steps=400)

            while viewer.is_running():
                running = controller.step_once()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.005)

                # 获取末端Z高度
                ee_pos = data.site_xpos[ee_site_id]
                ee_z = ee_pos[2]
                if ee_z > SAFE_HEIGHT:
                    print(f"⚠️ 高度警告！末端Z={ee_z:.3f} m 超过安全限制1.2 m，已停止此关节运动。")
                    break

                if not running:
                    print(f"✅ 第 {joint_id+1} 轴到达目标角度 {np.rad2deg(target_angle_rad):.1f}°")
                    print(f"   当前末端位置: X={ee_pos[0]:.3f}, Y={ee_pos[1]:.3f}, Z={ee_pos[2]:.3f}")
                    break

            # 每个 jog 后停顿 1.5 秒
            time.sleep(1.5)

        print("\n✅ Master calibration done — 零点标定完成 ✅")

        # 保持窗口打开观察
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
