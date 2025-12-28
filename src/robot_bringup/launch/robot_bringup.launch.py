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
            "pwm_hz": 200,
            "timeout_ms": 500,
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

    # --- NeoPixel UART ---
    neopixel = Node(
        package="neopixel_uart",
        executable="neopixel_uart_node",
        name="neopixel_uart_node",
        output="screen",
        parameters=[{
            "port": "/dev/ttyAMA0",
            "baud": 115200,
            "send_hz": 20,
            "log_frames": False,
        }],
    )

    # --- Servo driver ---
    servo = Node(
        package="servo_driver",
        executable="servo_driver_node",
        name="servo_driver_node",
        output="screen",
        parameters=[{
            "chip": "/dev/gpiochip4",
            "gpio": 12,
            "topic": "/servo/angle_deg",
            "min_us": 500,
            "max_us": 2500,
            "min_deg": 0.0,
            "max_deg": 180.0,
            "period_us": 20000,
            "max_step_us": 15,
            "rt_priority": 10,
            "hold_ms": 300,
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

    lane_mask = Node(
        package="lane_inference",
        executable="lane_mask_node",
        name="lane_mask_node",
        namespace="camera_bottom",
        output="screen",
        parameters=[{
            "in_topic": "/camera_bottom/camera_node/image_raw",
            "out_topic": "/camera_bottom/lane_mask",
            "model_path": "/home/robot/ros2_ws_robot/src/lane_inference/model/lane_lraspp_mbv3_best.pth",
            "img_w": 320,
            "img_h": 240,
            "thresh": 0.5,
            "rotate_deg": 180,
        }],
    )

    lane_center = Node(
        package="lane_inference",
        executable="lane_centerline_node",
        name="lane_centerline_node",
        namespace="camera_bottom",
        output="screen",
        parameters=[{
            "in_topic": "/camera_bottom/lane_mask",
            "out_topic": "/camera_bottom/center_lines",
            "info_topic": "/camera_bottom/lane_info",
            "skip_n": 0,
        }],
    )

    return LaunchDescription([
        bno08x,
        motor,
        camera_front,
        camera_bottom,
        neopixel,
        servo,
        lcd,
        audio,
        lane_mask,
        lane_center,
    ])