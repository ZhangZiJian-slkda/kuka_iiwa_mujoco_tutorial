'''
Author: Zhang-sklda 845603757@qq.com
Date: 2025-11-15 20:45:16
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-11-16 01:39:43
FilePath: /kuka_iiwa_mujoco_tutorial/trajectory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import mujoco.viewer
import ikpy.chain
import numpy as np
import transforms3d as tf #欧拉角→旋转矩阵
import time
import mujoco

def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:]=[0,0.5,0.5]
    viewer.cam.distance=2.5
    viewer.cam.elevation=-30
    viewer.cam.azimuth=180


class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""
    def __init__(self,start_joints,end_joints,steps,tol=0.02,max_hold=200):
        self.start_joints = np.array(start_joints,dtype=float)
        self.end_joints = np.array(end_joints,dtype=float)
        self.steps = max(1,int(steps))
        self.step = (self.end_joints-self.start_joints)/float(self.steps)
        self._gen = self._generate_trajectory()
        # self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints.copy()
        self.tol = float(tol)
        self._hold = 0
        self._max_hold = max_hold
        self._finished = False

    def _generate_trajectory(self):
        for i in range(self.steps+1):
            yield self.start_joints + self.step * i
            # 确保最后精确到达目标关节值
        yield self.end_joints

    def get_next_waypoint(self,qpos):
        # 检查当前的关节值是否已经接近目标路径点。若是，则更新下一个目标路径点；若否，则保持当前目标路径点不变。
        if np.allclose(qpos,self.waypoint,atol = self.tol):
            try:
                self.waypoint = next(self._gen)
                self._hold = 0
            except StopIteration:
                self._finished = True
                return self.waypoint.copy()
        else:
            self._hold += 1
            if self._hold >= self._max_hold:
                try:
                    self.waypoint = next(self._gen)
                    self._hold = 0
                except StopIteration:
                    self._finished = True
                    return self.waypoint.copy()
        return self.waypoint.copy()
    def finished(self):
        return self._finished
def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("kuka_iiwa_14/iiwa14.urdf",
                                               base_elements=["iiwa_link_0"])
    
    my_chain.active_links_mask[-1] = False  # 将末端固定 link 标记为非活动
    my_chain.active_links_mask[0] = False   # 将基座 link 标记为非活动
    print("链总长度:", len(my_chain.links))
    print("active_links_mask:", my_chain.active_links_mask)
    print("活动关节数量:", int(sum(my_chain.active_links_mask)))


    start_joints = np.array([0,0,0,1.57,0,-1.57,0])  # 起始关节角
    data.qpos[:7] = start_joints.copy()  # 初始化机械臂到起始位置

    target_position = [-0.34,0.3,0.4]  # 末端执行器在世界坐标系下的目标位置的xyz
    target_orientation = [0,0,0]  # Roll, Pitch, Yaw in radians
    rotation_matrix = tf.euler.euler2mat(*target_orientation)
    # initial_position = [0 ,0, 0, 0, 1.57, 0, 0, 0, 0]  # 提供初始关节角猜测，用于优化求解 
    #构造 initial_position（full length）
    full_len = len(my_chain.links)
    initial_full = [0]*full_len
    active_idx = [i for i, v in enumerate(my_chain.active_links_mask) if v]
    if len(active_idx) == len(start_joints):
        for idx, val in zip(active_idx, start_joints):
            initial_full[idx] = float(val)
    else:
        for idx in active_idx:
            initial_full[idx] = 0.0

    joint_angles_full  = my_chain.inverse_kinematics(
        target_position,
        target_orientation=rotation_matrix,
        initial_position=initial_full
    )
    # end_joints = joint_angles[:7]  # 提取机械臂的关节角
    print("joint_angles_full:", joint_angles_full)
    ctrl_joints = [joint_angles_full[i] for i, v in enumerate(my_chain.active_links_mask) if v ]
    end_joints = np.array(ctrl_joints)
    print("end_joints:", end_joints)

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=200,tol=0.01,max_hold=200)
    Kp = np.array([150.0, 120.0, 120.0, 80.0, 60.0, 50.0, 40.0])   # 7 元组
    Kd = np.array([5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 2.0])

    torque_limit = np.array([120.0]*7)  # 假设每个关节的力矩限制为 120 Nm

    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer_init(viewer)
        step =0
        last_time = time.time()
        while viewer.is_running():
            q = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            waypoint = joint_trajectory.get_next_waypoint(q)

            err = waypoint-q
            derr = -qd
            tau = Kp*err + Kd*derr
            tau = np.clip(tau,-torque_limit,torque_limit)
            data.ctrl[:7] = qd + tau

            mujoco.mj_step(model,data)
            viewer.sync()

            if step < 80 and step % 10 == 0:
                print(f"step {step} | waypoint: {np.round(waypoint,3)} | q: {np.round(q,3)} | tau: {np.round(tau,3)}")
            step += 1
            if joint_trajectory.finished() and np.allclose(q, end_joints, atol=0.02):
                print("Trajectory finished and reached goal.")
                break
    print("Simulation ended.")
if __name__ == "__main__":
    main()