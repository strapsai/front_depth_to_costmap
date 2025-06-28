from setuptools import setup
import os
from glob import glob

print("\n\n\n>>>>>> setup.py가 실행되고 있습니다! 이 메시지가 보이면 파일은 실행된 것입니다. <<<<<<\n\n\n")


package_name = 'depth_to_pointcloud_pub'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'transform3d', 
    ],
    zip_safe=True,
    maintainer='Minho Lee',
    maintainer_email='mhlee00@inha.edu',
    description='Convert depth+camera_info to PointCloud2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_to_pointcloud_node = depth_to_pointcloud_pub.depth_to_pointcloud_node:main'
        ],
    },
)
