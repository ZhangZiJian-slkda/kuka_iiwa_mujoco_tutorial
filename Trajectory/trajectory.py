'''
Author: Zhang-sklda 845603757@qq.com
Date: 2025-11-15 20:45:16
LastEditors: 张子健 16139863+abc845603757@user.noreply.gitee.com
LastEditTime: 2025-11-16 22:28:00
FilePath: /kuka_iiwa_mujoco_tutorial/trajectory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import mujoco.viewer
import numpy as np
import ikpy.chain
import transforms3d as tf

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
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        # 确保最后精确到达目标关节值
        yield self.end_joints

    def get_next_waypoint(self, qpos):
        # 检查当前的关节值是否已经接近目标路径点。若是，则更新下一个目标路径点；若否，则保持当前目标路径点不变。
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
    my_chain = ikpy.chain.Chain.from_urdf_file("kuka_iiwa_14/iiwa14.urdf",base_elements=["iiwa_link_0"])
    
    my_chain.active_links_mask[-1] = False  # 将末端固定 link 标记为非活动
    my_chain.active_links_mask[0] = False   # 将基座 link 标记为非活动

    start_joints = [0, 0, 0, 0, 0, 0, 0]
    data.qpos[1:8] = start_joints

    target_position = [-0.13, 0.6, 0.1]
    target_euler = [3.14, 0, 1.57]
    reference_position = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    target_orientation = tf.euler.euler2mat(*target_euler)

    joint_angles = my_chain.inverse_kinematics(target_position=target_position,target_orientation=target_orientation,reference_position=reference_position)
    end_joints = joint_angles[2:-1]

    joint_trajectory = JointSpaceTrajectory(start_joints,end_joints,steps=100)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:7])
            data.ctrl[:7]=waypoint
            mujoco.mj_step(model,data)
            viewer.sync()

if __name__ == "__main__":
    main()