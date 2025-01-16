import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

G1_DEX_USD_PATH = "/home/morning/IsaacLab/descriptions/g1_29dof_with_hand_v_1_0.usd"

G1_DEX_CFG_OLD = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_DEX_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)


G1_DEX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_DEX_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),  # initial position, set according to needs
        joint_pos={
            # leg joints
            "left_hip_pitch_joint": -0.20,
            "left_hip_roll_joint": 0.16,
            "left_hip_yaw_joint": 0.0,  # set according to needs
            "left_knee_joint": 0.42,
            "left_ankle_pitch_joint": -0.23,
            "left_ankle_roll_joint": 0.0,  # set according to needs

            "right_hip_pitch_joint": -0.20,
            "right_hip_roll_joint": -0.16,
            "right_hip_yaw_joint": 0.0,  # set according to needs
            "right_knee_joint": 0.42,
            "right_ankle_pitch_joint": -0.23,
            "right_ankle_roll_joint": 0.0,  # set according to needs

            # body joints
            "waist_yaw_joint": 0.0,  # set according to needs
            # "waist_roll_joint": 0.0,  # fixed joint
            # "waist_pitch_joint": 0.0,  # fixed joint
            # "head_joint": 0.0,         # fixed joint

            # left arm joints
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_yaw_joint": 0.0,  # set according to needs
            "left_elbow_joint": 0.87,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            # right arm joints
            "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_yaw_joint": 0.0,  # set according to needs
            "right_elbow_joint": -0.87,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,

            # hand joints
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,

            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
        },
        joint_vel={
            ".*": 0.0,  # initial speed for all joints
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_yaw_joint",
                "left_hip_roll_joint",
                "left_hip_pitch_joint",
                "left_knee_joint",
                "right_hip_yaw_joint",
                "right_hip_roll_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "left_hip_yaw_joint": 150.0,
                "left_hip_roll_joint": 150.0,
                "left_hip_pitch_joint": 200.0,
                "left_knee_joint": 200.0,
                "right_hip_yaw_joint": 150.0,
                "right_hip_roll_joint": 150.0,
                "right_hip_pitch_joint": 200.0,
                "right_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                "left_hip_yaw_joint": 5.0,
                "left_hip_roll_joint": 5.0,
                "left_hip_pitch_joint": 5.0,
                "left_knee_joint": 5.0,
                "right_hip_yaw_joint": 5.0,
                "right_hip_roll_joint": 5.0,
                "right_hip_pitch_joint": 5.0,
                "right_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
            },
            armature={
                "left_hip_yaw_joint": 0.01,
                "left_hip_roll_joint": 0.01,
                "left_hip_pitch_joint": 0.01,
                "left_knee_joint": 0.01,
                "right_hip_yaw_joint": 0.01,
                "right_hip_roll_joint": 0.01,
                "right_hip_pitch_joint": 0.01,
                "right_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "left_shoulder_pitch_joint": 0.01,
                "left_shoulder_roll_joint": 0.01,
                "left_shoulder_yaw_joint": 0.01,
                "left_elbow_joint": 0.01,
                "left_wrist_roll_joint": 0.01,
                "left_wrist_pitch_joint": 0.01,
                "left_wrist_yaw_joint": 0.01,
                "right_shoulder_pitch_joint": 0.01,
                "right_shoulder_roll_joint": 0.01,
                "right_shoulder_yaw_joint": 0.01,
                "right_elbow_joint": 0.01,
                "right_wrist_roll_joint": 0.01,
                "right_wrist_pitch_joint": 0.01,
                "right_wrist_yaw_joint": 0.01,
            },
        ),
        # set hand actuator according to needs
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
            ],
            effort_limit=10,
            velocity_limit=50.0,
            stiffness=30.0,
            damping=5.0,
            armature=0.005,
        ),
    },
)