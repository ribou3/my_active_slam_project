from setuptools import setup
import os

package_name = 'active_slam_exploration'

setup(
    name=package_name,
    version='0.0.0',
    # 実際に存在するディレクトリをすべてリストアップ
    packages=[package_name, 'examples', 'examples.turtlebot', 'gym_gazebo'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ribou',
    description='ROS 2 Humble port',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'test_reset = examples.turtlebot.test_reset:main',
        ],
    },
)
