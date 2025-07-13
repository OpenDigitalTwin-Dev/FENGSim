#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import launch.conditions

def generate_launch_description():
    # Get the shared directory of the package
    pkg_share = get_package_share_directory('two_dof_arm')

    # Declare startup parameters
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='circle',
        description='Target publishing mode: circle, manual, or interactive'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start rviz'
    )

    urdf_file = os.path.join(pkg_share, 'urdf', 'two_dof_arm.urdf')

    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()
    
    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': False
        }]
    )

    # Joint state publisher gui node
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('use_custom_controller', default='true'))
    )

    # Robotic arm control node
    arm_node = Node(
        package='two_dof_arm',
        executable='two_dof_arm_node',
        name='two_dof_arm',
        output='screen',
        parameters=[{
            'use_sim_time': False,
        }]
    )

    # Target publish node
    target_node = Node(
        package='two_dof_arm',
        executable='target_publisher.py',
        name='target_publisher',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode')
        }]
    )

    # rviz node
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'two_dof_arm.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_rviz'))
    )


    # TF publisher from world to base_link
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
    )

    return LaunchDescription([
        mode_arg,
        use_rviz_arg,
        robot_state_publisher_node,
        arm_node,
        target_node,
        rviz_node,
        static_tf
    ])