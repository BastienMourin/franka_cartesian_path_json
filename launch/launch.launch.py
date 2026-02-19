from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import math
from launch_ros.actions import Node

def generate_nodes(context):
    nodes = []

    # ---- Robot 1: NS1 ----
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py'
                ])
            ),
            launch_arguments={
                'robot_ip': '172.16.2.2',
                'namespace': 'NS1',
                'use_sim_time': 'false',   # optional, depends on your setup
                'robot_name': 'NS1',       # ensures moveit parameters are namespaced
            }.items()
        )
    )

    # ---- Robot 2: NS2 ----
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('franka_fr3_moveit_config'), 'launch', 'moveit.launch.py'
                ])
            ),
            launch_arguments={
                'robot_ip': '172.16.3.2',
                'namespace': 'NS2',
                'use_sim_time': 'false',
                'robot_name': 'NS2',
            }.items()
        )
    )

    # ---- Static transform to place NS2 relative to NS1 ----
    tx = 1.6  # meters
    ty = 0.0
    tz = 0.02
    yaw = math.pi  # 180 deg
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    nodes.append(
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_NS1_to_NS2',
            arguments=[
                str(tx), str(ty), str(tz),
                str(qx), str(qy), str(qz), str(qw),
                'NS1_base', 'NS2_base'
            ],
            output='screen',
        )
    )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=generate_nodes)
    ])

