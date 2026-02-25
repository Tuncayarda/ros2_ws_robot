from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # --- BNO08X ---
    bno_pkg_share = get_package_share_directory("bno08x_driver")
    bno_launch = os.path.join(bno_pkg_share, "launch", "bno085_i2c.launch.py")

    bno08x = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bno_launch)
    )

    # --- Motor driver ---
    motor = Node(
        package="motor_driver",
        executable="motor_driver_node",
        name="motor_driver_node",
        output="screen",
        parameters=[{
            "left_fwd": 26,
            "left_rev": 13,
            "right_fwd": 16,
            "right_rev": 24,
            "gpiochip": "gpiochip4",
            "pwm_hz": 100,
            "timeout_ms": 1000,
            "invert_left": False,
            "invert_right": False,
        }],
        remappings=[
            ("/motor_cmd", "/motor_cmd"),
        ],
    )

    # --- Front camera ---
    camera_front = Node(
        package="camera_ros",
        executable="camera_node",
        name="camera_node",
        namespace="camera_front",
        output="screen",
        parameters=[{
            "camera": 0,
            "role": "viewfinder",
            "format": "BGR888",
        }],
    )

    # --- Bottom camera ---
    camera_bottom = Node(
        package="camera_ros",
        executable="camera_node",
        name="camera_node",
        namespace="camera_bottom",
        output="screen",
        parameters=[{
            "camera": 1,
            "role": "viewfinder",
            "format": "BGR888",
            "width": 320,
            "height": 240,
        }],
    )

    # --- Waveshare LCD ---
    lcd = Node(
        package="waveshare_lcd",
        executable="lcd_node",
        name="lcd_node",
        output="screen",
        parameters=[{
            "fps": 20,
            "loop": False,
            "roi_enable": True,
            "roi_pad": 2,
            "spi_hz": 20000000,
            "spi_chunk": 32768,
            "spi_use_multi_ioc": False,
            "rotate_cw_90": True,
        }],
    )

    audio = Node(
        package="audio_player",
        executable="audio_player_node",
        name="audio_player_node",
        output="screen",
        parameters=[{
            "topic_path": "/audio/path",
            "topic_url":  "/audio/url",
            "topic_stop": "/audio/stop",
            "player": "mpv",
            "volume": 20,
            "ytdl": True,
            "alsa_device": "",
        }],
    )

    demo_core = Node(
        package="demo_core",
        executable="demo_core_node",
        name="demo_core",
        output="screen",
        parameters=[{
            "sounds_dir": "/home/robot/Sounds",
            "audio_topic": "/audio/path",
            "servo_topic": "/servo/angle_deg",
        }],
    )

    # lane_mask = Node(
    #     package="lane_inference",
    #     executable="lane_mask_node",
    #     name="lane_mask_node",
    #     namespace="camera_bottom",
    #     output="screen",
    #     parameters=[{
    #         "in_topic": "/camera_bottom/camera_node/image_raw",
    #         "out_topic": "/camera_bottom/lane_mask",
    #         "model_path": "/home/robot/ros2_ws_robot/src/lane_inference/model/lane_lraspp_mnv3_best_iou.pth",
    #         "img_w": 320,
    #         "img_h": 240,
    #         "thresh": 0.5,
    #         "rotate_deg": 180,
    #     }],
    # )

    # lane_center = Node(
    #     package="lane_centerline_cpp",
    #     executable="lane_centerline_node",
    #     name="lane_centerline_node",
    #     namespace="camera_bottom",
    #     output="screen",
    # )

    # --- rosbridge websocket (mobile/pc <-> ROS2) ---
    rosbridge = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            "port": 9090,
            "address": "0.0.0.0",
            "delay_between_messages": 0.0,
        }],
    )

    # --- Mobile COM (HTTP upload server for mp3) ---
    mobile_com = Node(
        package="mobile_com",
        executable="mobile_com_node",
        name="mobile_com",
        output="screen",
        parameters=[{
            "host": "0.0.0.0",
            "port": 8000,
            "sounds_dir": "/home/robot/Sounds",  # burada kaydedecek
            "max_upload_mb": 50,                 # 8MB rahat, istersen 100 yap
        }],
    )

    # --- Pico COM (LED TX + Sensor RX + Servo) ---
    pico_com = Node(
        package="pico_com",
        executable="pico_com",
        name="pico_com",
        output="screen",
        parameters=[{
            "port": "/dev/ttyAMA0",
            "baud": 115200,
            "send_hz": 20,
            "rx_hz": 200,
            "frame_timeout_ms": 80,
            "log_frames": False,
            "publish_raw": False,
            "servo_topic": "/servo/angle_deg",
            "servo_center_deg": 134.0,
            "servo_min_deg": 94.0,
            "servo_max_deg": 174.0,
        }],
    )

    # --- Battery ADS1115 ---
    battery = Node(
        package="battery_manager",
        executable="battery_manager_node",
        name="battery_ads1115_node",
        output="screen",
        parameters=[{
            "i2c_bus": 1,
            "i2c_addr": 72,
            "divider_ratio": 5.0,
            "publish_hz": 2.0,
            "ema_alpha": 0.3,
        }],
        remappings=[
            ("/battery_status", "/battery_status"),
        ],
    )

    return LaunchDescription([
        bno08x,
        motor,
        camera_front,
        camera_bottom,
        lcd,
        audio,
        demo_core,
        # lane_mask,
        # lane_center,
        rosbridge,
        mobile_com,
        pico_com,
        battery,
    ])