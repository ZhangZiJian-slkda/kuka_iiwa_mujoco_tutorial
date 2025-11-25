# kuka_iiwa_mujoco_tutorial
Mujoco example based on kuka-iiwa
======================================================
KUKA iiwa MuJoCo 仿真教程
基于 MuJoCo 的 KUKA iiwa 机械臂仿真项目，从零搭建完整的机器人仿真实验环境。

项目简介
本项目参考 MuJoCo 全流程实战教程
https://github.com/VincentXun/tutorial_for_mujoco.git
将原教程中的 UR5e 机械臂替换为 KUKA iiwa 14 机械臂，实现了完整的机器人建模、运动学计算,通过IKPy库求解逆运动学，控制机械臂运动,添加触觉传感器与被推动物体的模型。

核心功能
KUKA iiwa 14 模型 - 完整的 7 自由度工业机械臂
正向运动学 - 精确的末端执行器位置计算
逆运动学求解 - 使用 IKPy 库进行逆运动学计算
轨迹规划 - 关节空间和笛卡尔空间轨迹生成
阻抗控制 - 力控和柔顺控制算法(未完成-待后续)
触觉传感器 - 接触检测和力反馈
物体交互 - 推动物体和场景交互

kuka_iiwa_mujoco_tutorial/
├── kuka_iiwa_14/                 # KUKA iiwa 14 机械臂模型
│   ├── assets/                   # 3D 模型资源
│   │   ├── *.obj                 # 机械臂部件模型文件
│   │   └── band.obj              # 传送带模型
│   ├── iiwa14.xml               # MuJoCo 模型文件
│   ├── iiwa14.urdf              # URDF 模型文件
│   ├── scene.xml                # 场景配置文件
│   └── README.md                # 模型说明文档
├── IKPY_Test/                   # 逆运动学测试
│   ├── main.py                  # 基础程序
│   ├── ikMujoco.py              # MuJoCo 逆运动学实现
│   └── ikTest.py                # 逆运动学测试
├── Trajectory/                  # 轨迹规划
│   └── trajectory.py            # 轨迹生成算法
├── ImpedanceControl/            # 阻抗控制
│   └── (阻抗控制相关代码)
├── requirements.txt             # 项目依赖
└── README.md                   # 项目说明

环境配置
1. 创建 Conda 环境
bash
conda create -n kuka_iiwa_mujoco python=3.9
conda activate kuka_iiwa_mujoco
2. 安装依赖库
bash
pip install -r requirements.txt
3. 验证安装
bash
python -c "import mujoco; print('MuJoCo 安装成功')"


