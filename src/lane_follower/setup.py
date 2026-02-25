import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lane_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='tncyard@gmail.com',
    description='Lane follower: detector + controller nodes',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_follower = lane_follower.lane_follower_node:main',
            'lane_detector = lane_follower.lane_detector_node:main',
            'lane_controller = lane_follower.lane_controller_node:main',
        ],
    },
)
