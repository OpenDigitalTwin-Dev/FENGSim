from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包的共享目录
    pkg_share = get_package_share_directory('additive_manufacturing_sim')
    
    # URDF文件路径
    urdf_file = os.path.join(pkg_share, 'urdf', 'printer.urdf')
    
    # RViz配置文件路径
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'additive_sim.rviz')
    
    # 检查文件是否存在
    if not os.path.exists(urdf_file):
        print(f"Warning: URDF file not found at {urdf_file}")
    if not os.path.exists(rviz_config_file):
        print(f"Warning: RViz config file not found at {rviz_config_file}")
    
    # 读取URDF文件内容
    with open(urdf_file, 'r') as infp:
        robot_description_content = infp.read()
    
    # 声明启动参数
    print_mode_arg = DeclareLaunchArgument(
        'print_mode',
        default_value='custom',
        description='Print mode: square, circle, or custom'
    )
    
    print_speed_arg = DeclareLaunchArgument(
        'print_speed',
        default_value='0.05',
        description='Print head movement speed (m/s)'
    )
    
    layer_height_arg = DeclareLaunchArgument(
        'layer_height',
        default_value='0.002',
        description='Layer height (m)'
    )
    
    num_layers_arg = DeclareLaunchArgument(
        'num_layers',
        default_value='50',
        description='Number of layers to print'
    )
    
    # Robot State Publisher节点
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
    
    # Joint State Publisher节点
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'use_gui': False,
            'rate': 50
        }]
    )
    
    # 增材制造仿真节点
    additive_sim_node = Node(
        package='additive_manufacturing_sim',
        executable='additive_sim_node',
        name='additive_sim_node',
        output='screen',
        parameters=[{
            'print_mode': LaunchConfiguration('print_mode'),
            'print_speed': LaunchConfiguration('print_speed'),
            'layer_height': LaunchConfiguration('layer_height'),
            'num_layers': LaunchConfiguration('num_layers')
        }]
    )
    
    # RViz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )
    
    # 创建并返回launch描述
    return LaunchDescription([
        # 声明参数
        print_mode_arg,
        print_speed_arg,
        layer_height_arg,
        num_layers_arg,
        # 启动节点
        robot_state_publisher_node,
        joint_state_publisher_node,
        additive_sim_node,
        rviz_node
    ])