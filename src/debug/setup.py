from setuptools import find_packages, setup

package_name = 'debug'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='tncyard@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'imu_motion_test = debug.imu_motion_test:main',
            'motor_calibration_node = debug.motor_calibration_node:main',
            'straight_drive_node = debug.straight_drive_node:main',
            'camera_sampler = debug.camera_sampler_node:main',
            'lane_mask_jpegpipe_oneshot = debug.frame_test:main',
        ],
    },
)
