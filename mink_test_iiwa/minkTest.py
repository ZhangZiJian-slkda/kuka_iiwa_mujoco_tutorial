"""
Description: Robotic Arm Motion Control Algorithm
Author: Zhang-sklda 845603757@qq.com
Date: 2025-12-04 23:05:46
Version: 1.0.0
LastEditors: Zhang-sklda 845603757@qq.com
LastEditTime: 2025-12-06 00:01:11
FilePath: /kuka_iiwa_mujoco_tutorial/mink_test_iiwa/minkTest.py
Copyright (c) 2025 by Zhang-sklda, All Rights Reserved.
symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
"""
# """
# Description: Robotic Arm Motion Control Algorithm
# Author: Zhang-sklda 845603757@qq.com
# Date: 2025-12-04 23:05:46
# Version: 1.0.0
# LastEditors: Zhang-sklda 845603757@qq.com
# LastEditTime: 2025-12-04 23:32:58
# FilePath: /kuka_iiwa_mujoco_tutorial/mink_test_iiwa/minkTest.py
# Copyright (c) 2025 by Zhang-sklda, All Rights Reserved.
# symbol_custom_string_obkoro1_tech: Tech: Motion Control | MuJoCo | ROS | Kinematics
# """
import mujoco
import mujoco.viewer
import numpy as np
import mink
from loop_rate_limiters import RateLimiter
def main():
    model = mujoco.MjModel.from_xml_path("mink_test_iiwa/kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    # setup ik solver
    configuration = mink.Configuration(model) 
    tasks = [end_effector_task := mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1.0,
    ),
    posture_task := mink.PostureTask(model,cost = 1e-4),
    ]


    # IK settings
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20


    with mujoco.viewer.launch_passive(model, data,show_left_ui=False,show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data,model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # mink.move_mocap_to_frame(model=model,data=data,frame_name="target",attach_site_name="attachment_site",site_name = "site")
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        rate = RateLimiter(frequency=500.0,warn=False)
        # dt = 0.01

        while viewer.is_running():
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            for i in range (max_iters):
                vel = mink.solve_ik(configuration,tasks,rate.dt,solver,1e-3)
                configuration.integrate_inplace(vel,rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break
            data.ctrl = configuration.q
            mujoco.mj_step(model, data)
            viewer.sync()
            # rate.sleep()

if __name__ == "__main__":
    main()