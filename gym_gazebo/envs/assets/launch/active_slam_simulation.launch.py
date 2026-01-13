# import os
# from ament_index_python.packages import get_package_share_directory

# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node


# def generate_launch_description():
#     # -------------------------
#     # 共通設定
#     # -------------------------
#     use_sim_time = LaunchConfiguration('use_sim_time', default='true')

#     world_file = os.path.expanduser(
#         '~/ros2_ws/src/my_active_slam_project/gym_gazebo/envs/assets/worlds/circuit2.world'
#     )

#     # -------------------------
#     # 1. Gazebo 起動
#     # -------------------------
#     gazebo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(
#                 get_package_share_directory('gazebo_ros'),
#                 'launch',
#                 'gazebo.launch.py'
#             )
#         ),
#         launch_arguments={'world': world_file}.items(),
#     )

#     # -------------------------
#     # 2. TurtleBot3 spawn（SDF）
#     # -------------------------
#     spawn_robot = Node(
#         package='gazebo_ros',
#         executable='spawn_entity.py',
#         arguments=[
#             '-entity', 'tb3',
#             '-file',
#             '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
#             '-x', '0.5',
#             '-y', '0.5',
#             '-z', '0.01'
#         ],
#         output='screen'
#     )

#     # -------------------------
#     # 3. SLAM Toolbox
#     # -------------------------
#     slam = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(
#                 get_package_share_directory('slam_toolbox'),
#                 'launch',
#                 'online_async_launch.py'
#             )
#         ),
#         launch_arguments={'use_sim_time': use_sim_time}.items(),
#     )

#     # -------------------------
#     # 4. Teleop
#     # -------------------------
#     teleop = Node(
#         package='teleop_twist_keyboard',
#         executable='teleop_twist_keyboard',
#         prefix='xterm -e',
#         output='screen'
#     )

#     return LaunchDescription([
#         SetEnvironmentVariable(
#             name='TURTLEBOT3_MODEL',
#             value='burger'
#         ),
#         gazebo,
#         spawn_robot,
#         slam,
#         teleop,
#     ])


import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # -------------------------
    # 共通設定
    # -------------------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # world ファイル（package share 推奨）
    world_file = os.path.join(
        get_package_share_directory('active_slam_exploration'),
        'gym_gazebo',
        'envs',
        'assets',
        'worlds',
        'circuit2.world'
    )


    # -------------------------
    # 1. Gazebo 起動
    # -------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items(),
    )

    # -------------------------
    # 2. TurtleBot3 Robot State Publisher（最重要）
    # -------------------------
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_bringup'),
                'launch',
                'robot_state_publisher.launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items(),
    )

    # -------------------------
    # 3. TurtleBot3 spawn（Gazebo 完全起動後）
    # -------------------------
    spawn_robot = TimerAction(
        period=3.0,  # Gazebo 起動待ち
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'tb3',
                    '-file',
                    os.path.join(
                        get_package_share_directory('turtlebot3_gazebo'),
                        'models',
                        'turtlebot3_burger',
                        'model.sdf'
                    ),
                    '-x', '0.5',
                    '-y', '0.5',
                    '-z', '0.01'
                ],
                output='screen'
            )
        ]
    )

    # -------------------------
    # 4. SLAM Toolbox
    # -------------------------
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items(),
    )

    # -------------------------
    # 5. Teleop（※学習時はコメントアウト推奨）
    # -------------------------
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        prefix='xterm -e',
        output='screen'
    )

    # -------------------------
    # Launch Description
    # -------------------------
    return LaunchDescription([
        # TurtleBot3 モデル指定（必須）
        SetEnvironmentVariable(
            name='TURTLEBOT3_MODEL',
            value='burger'
        ),

        gazebo,
        robot_state_publisher,
        spawn_robot,
        slam,
        teleop,  # ← 学習時は外す
    ])
