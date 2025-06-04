from setuptools import setup

package_name = 'depth_to_pointcloud_pub'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/depth_to_pointcloud.launch.py']),
        ('share/' + package_name + '/config', ['config/frontleft_info.yaml', 'config/frontright_info.yaml']),
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
