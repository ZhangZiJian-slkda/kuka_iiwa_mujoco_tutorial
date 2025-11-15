'''
Author: Zhang-sklda 845603757@qq.com
Date: 2025-11-15 16:18:37
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-11-15 18:58:44
FilePath: /kuka_iiwa_mujoco_tutorial/ikTest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d as tf

def main():
    my_chain = ikpy.chain.Chain.from_urdf_file("kuka_iiwa_14/iiwa14.urdf",
                                               base_elements=["iiwa_link_0"])
    
    my_chain.active_links_mask[-1] = False  # 将末端固定 link 标记为非活动
    my_chain.active_links_mask[0] = False   # 将基座 link 标记为非活动
    # print("链总长度:", len(my_chain.links))
    # print("active_links_mask:", my_chain.active_links_mask)
    # print("活动关节数量:", sum(my_chain.active_links_mask))

    target_position = [0.5, 0, 0.5]  # 末端执行器在世界坐标系下的目标位置的xyz
    target_orientation = [0 ,0,0]  # Roll, Pitch, Yaw in radians
    rotation_matrix = tf.euler.euler2mat(*target_orientation)
    initial_position = [0 ,0, 0, 0, 1.57, 0, 0, 3.0, 0]  # 提供初始关节角猜测，用于优化求解

    fig, ax = plot_utils.init_3d_figure()
    # my_chain.plot(my_chain.inverse_kinematics(target_position,rotation_matrix,initial_position))
    joint_angles = my_chain.inverse_kinematics(
        target_position,
        target_orientation=rotation_matrix,   # 用关键字参数
        initial_position=initial_position     # 用关键字参数
    )
    my_chain.plot(joint_angles, ax)
    plt.show()

    
if __name__ == "__main__":
    main()
