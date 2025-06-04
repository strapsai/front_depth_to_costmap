from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_to_pointcloud_pub',
            executable='depth_to_pointcloud_node',
            name='depth_to_pointcloud_node',
            output='screen',
            emulate_tty=True
        )
    ])
