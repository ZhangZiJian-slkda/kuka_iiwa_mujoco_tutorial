'''
Author: Zhang-sklda 845603757@qq.com
Date: 2025-11-15 20:13:11
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-11-15 20:43:58
FilePath: /kuka_iiwa_mujoco_tutorial/ikMujoco.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import mujoco.viewer
import time 
import ikpy.chain
import transforms3d as tf

def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("kuka_iiwa_14/iiwa14.urdf",
                                               base_elements=["iiwa_link_0"])
    
    my_chain.active_links_mask[-1] = False  # 将末端固定 link 标记为非活动
    my_chain.active_links_mask[0] = False   # 将基座 link 标记为非活动

    target_position = [-0.13,0.3,0.1]  # 末端执行器在世界坐标系下的目标位置的xyz
    target_orientation = [0 ,0, 0]  # Roll, Pitch, Yaw in radians
    rotation_matrix = tf.euler.euler2mat(*target_orientation)
    initial_position = [0 ,0, 0, 0, 1.57, 0, 1.57, 0, 0]  # 提供初始关节角猜测，用于优化求解 

    ee_id = model.site("attachment_site").id

    joint_angles = my_chain.inverse_kinematics(
        target_position,
        target_orientation=rotation_matrix,   # 用关键字参数
        initial_position=initial_position     # 用关键字参数
    )
    ctrl = joint_angles[1:-1]
    data.ctrl[:7] = ctrl

    with mujoco.viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

    target_position= data.site_xpos[ee_id]
    print("机械臂末端执行器位置：", target_position)
    
if __name__ == "__main__":
    main()
