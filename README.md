# ROS 2 Robot Workspace

This repository is a ROS 2 workspace that powers a mobile robot with cameras, motors, audio playback, LCD display, Pico telemetry, battery monitoring, and a manual demo core. The bringup is centralized in `robot_bringup` and launches the system as a single stack.

## Highlights

- **Robot bringup** that launches all core nodes.
- **Motor driver** with PWM and watchdog safety.
- **Servo driver** for a single PWM servo (software PWM on GPIO).
- **Audio player** based on `mpv` (file path or URL playback).
- **Mobile communication** server to upload sounds and emojis.
- **Pico UART bridge** for LED control and sensor telemetry.
- **Battery ADS1115** voltage + SOC estimation.
- **Waveshare LCD** image/video display over SPI.
- **Demo core**: manual show mode triggered from mobile topics.
- **Lane inference** (optional) with PyTorch segmentation.

## Workspace layout

```
ros2_ws_robot/
  src/
    audio_player/        # mpv-based audio playback node
    battery_manager/     # ADS1115 battery node
    bno08x-ros2-driver/   # IMU driver (external)
    camera_ros/          # libcamera ROS2 node (external)
    demo_core/           # manual demo logic
    lane_centerline_cpp/ # lane centerline (optional)
    lane_inference/      # segmentation model (optional)
    mobile_com/          # HTTP uploads + ROS topics
    motor_driver/        # dual H-bridge motor driver
    pico_com/            # UART bridge to Pico
    robot_bringup/       # launch files
    servo_driver/        # software PWM servo driver
    waveshare_lcd/       # SPI LCD display
```

## Bringup

Main launch file:
- `robot_bringup/launch/robot_bringup.launch.py`

It launches:
- **IMU**: `bno08x_driver`
- **Motor**: `motor_driver_node`
- **Cameras**: `camera_ros` (front and bottom)
- **Servo**: `servo_driver_node`
- **LCD**: `lcd_node`
- **Audio**: `audio_player_node`
- **Demo core**: `demo_core_node`
- **ROSBridge**: `rosbridge_websocket`
- **Mobile COM**: `mobile_com_node`
- **Pico COM**: `pico_com`
- **Battery**: `battery_manager_node`

## Demo core (manual show mode)

The demo core listens for single-shot triggers. When a message with `data = 1` arrives on any topic, the corresponding action is triggered.

### Topics

**Segment categories**
- `/segment/battery`
- `/segment/plastic`
- `/segment/metal`
- `/segment/glass`
- `/segment/paper`

**Slogan & analyze**
- `/robot/slogan`
- `/robot/analyze`

### Sound naming

Sounds are read from `/home/robot/Sounds` by default.

- Segment sounds must start with: `segment-<category>`
  - Example: `segment-battery-1.mp3`, `segment-battery-2.mp3`
- Slogan sound: `slogan.mp3`
- Analyze sound: `analyze.mp3` (also accepts `analyse.mp3`)

### Analyze behavior

When `/robot/analyze` is triggered:
- `analyze.mp3` plays
- Servo is commanded to `+40` degrees offset, then `-40` degrees offset

## Core topics summary

### Motor driver (`motor_driver`)
- **Sub**: `/motor_cmd` (`std_msgs/Int16MultiArray`) → `[left, right]` in -1000..1000

### Servo driver (`servo_driver`)
- **Sub**: `/servo/angle_deg` (`std_msgs/Float32`) → offset in degrees (e.g. -40..+40)

### Audio player (`audio_player`)
- **Sub**: `/audio/path` (`std_msgs/String`) → local file path
- **Sub**: `/audio/url` (`std_msgs/String`) → URL
- **Sub**: `/audio/stop` (`std_msgs/Empty`)
- **Sub**: `/audio/volume` (`std_msgs/Int32`)
- **Sub**: `/audio/volume/get` (`std_msgs/String`) → request volume state
- **Pub**: `/audio/volume_state` (`std_msgs/Int32`)

### Battery manager (`battery_manager`)
- **Pub**: `/battery_status` (`std_msgs/Float32MultiArray`)
  - index `0`: SOC (%)
  - index `1`: voltage (V)

### Pico COM (`pico_com`)
- **Sub**: `/led/mode` (`std_msgs/UInt8`)
- **Sub**: `/led/speed_ms` (`std_msgs/UInt16`)
- **Sub**: `/led/brightness` (`std_msgs/UInt8`)
- **Sub**: `/led/colors` (`std_msgs/UInt8MultiArray`)
- **Pub**: `/pico/telem/flags` (`std_msgs/UInt8`)
- **Pub**: `/pico/ccs811/eco2_ppm` (`std_msgs/UInt16`)
- **Pub**: `/pico/ccs811/tvoc_ppb` (`std_msgs/UInt16`)
- **Pub**: `/pico/dht22/temp_c_x100` (`std_msgs/Int16`)
- **Pub**: `/pico/dht22/hum_pct_x100` (`std_msgs/UInt16`)
- **Pub**: `/pico/mic/rms` (`std_msgs/UInt16`)
- **Pub**: `/pico/mic/peak` (`std_msgs/UInt16`)

### LCD (`waveshare_lcd`)
- **Sub**: `/lcd/media_path` (`std_msgs/String`) → file path (image/video)
- **Sub**: `/lcd/image` (`sensor_msgs/Image`)

## Mobile COM (uploads)

The mobile server exposes:

- `GET /health`
- `POST /upload_mp3` → upload sound into `/home/robot/Sounds`
- `POST /upload_emoji` → upload emoji into `/home/robot/Emojis`

ROS topics for list/delete:
- `/mobile/sounds/list_req`, `/mobile/sounds/list_res`
- `/mobile/sounds/delete_req`, `/mobile/sounds/delete_res`
- `/mobile/emojis/list_req`, `/mobile/emojis/list_res`
- `/mobile/emojis/delete_req`, `/mobile/emojis/delete_res`

## Optional lane inference

`lane_inference` contains a PyTorch segmentation node (`lane_mask_node`) that can be enabled in the bringup. It subscribes to a camera image, runs inference, and publishes a mono mask image.

## Build

Typical ROS 2 build (from workspace root):

```
colcon build --symlink-install
source install/setup.bash
```

## Run

```
ros2 launch robot_bringup robot_bringup.launch.py
```

## Notes

- `audio_player` requires `mpv` to be installed on the system.
- `camera_ros` depends on `libcamera`.
- `servo_driver` and `motor_driver` access GPIO via `gpiod`.
- `battery_manager` reads ADS1115 via I2C (`smbus2`).

## License

This workspace aggregates multiple packages; check each package’s `package.xml` and upstream licenses for details.
