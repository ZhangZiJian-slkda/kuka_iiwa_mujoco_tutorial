import mujoco.viewer
import time

def main():
    # Load and display the MuJoCo model
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    data.ctrl[:7] = [0, 0, 0, -1.57, 0, 0, 3.14]  # Set initial joint positions

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

if __name__ == "__main__":
    main()