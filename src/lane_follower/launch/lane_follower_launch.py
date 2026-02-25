"""
lane_follower launch
────────────────────
2 node başlatır:
  1) lane_detector_node  → görüntü işleme, /lane/detection ve /lane/debug_image publish
  2) lane_controller_node → motor kontrol, kesişim yönetimi

Sadece detector çalıştırıp debug image'e bakarak parametre ayarlamak:
  ros2 launch lane_follower lane_follower_launch.py detector_only:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Launch argümanları ──
    detector_only_arg = DeclareLaunchArgument(
        "detector_only", default_value="false",
        description="true ise sadece detector çalışır, motor sürülmez"
    )

    detector_only = LaunchConfiguration("detector_only")

    # ══════════════════════════════════════════
    # DETECTOR NODE
    # ══════════════════════════════════════════
    detector_node = Node(
        package="lane_follower",
        executable="lane_detector",
        name="lane_detector_node",
        output="screen",
        parameters=[{
            # Topics
            "image_topic": "/camera_bottom/camera_node/image_raw",
            "detection_topic": "/lane/detection",
            "debug_image_topic": "/lane/debug_image",
            "publish_debug_image": True,

            # ROI
            "roi_height_ratio": 0.40,
            "rotate_180": True,

            # Beyaz şerit algılama
            "blur_ksize": 5,
            "clahe_clip": 4.0,
            "adaptive_block": 31,
            "adaptive_c": -8,
            "sat_max": 255,
            "val_min": 10,

            # Kontur filtreleme
            "min_contour_area": 200,
            "min_aspect_ratio": 0.8,
            "morph_kernel": 5,

            # Kesişim eşikleri
            "intersection_white_ratio": 0.30,
            "intersection_width_ratio": 0.70,
        }],
    )

    # ══════════════════════════════════════════
    # CONTROLLER NODE
    # ══════════════════════════════════════════
    controller_node = Node(
        package="lane_follower",
        executable="lane_controller",
        name="lane_controller_node",
        output="screen",
        condition=UnlessCondition(detector_only),
        parameters=[{
            # Topics
            "detection_topic": "/lane/detection",
            "motor_topic": "/motor_cmd",
            "turn_cmd_topic": "/lane/turn_cmd",
            "status_topic": "/lane/status",

            # Motor kontrol
            "base_speed": 140,
            "kp": 200.0,
            "ki": 0.0,
            "kd": 40.0,
            "max_speed": 300,
            "max_turn": 140,
            "err_deadband_px": 6,
            "turn_smoothing": 0.3,
            "invert_steering": True,
            "camera_offset_px": 0,

            # Kesişim
            "intersection_confirm_frames": 4,
            "reverse_speed": 120,
            "reverse_duration": 0.6,

            # Dönüş manevrası (ileri-sağ/ileri-sol)
            "turn_speed_inner": 60,
            "turn_speed_outer": 220,
            "turn_duration": 1.5,
            "forward_after_turn": 0.4,
        }],
    )

    return LaunchDescription([
        detector_only_arg,
        detector_node,
        controller_node,
    ])
