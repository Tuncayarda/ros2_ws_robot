#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Imu


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class LaneSample:
    t: float
    dx_norm: Optional[float]   # [-1..+1]


@dataclass
class ImuSample:
    t: float
    yaw: float
    wz: float


class LaneFollowImuHold(Node):
    """
    Single-lane tracking with IMU heading hold.
    - IMU is primary (keeps heading stable)
    - Vision provides small correction (trim) via dx_norm
    - No turns, no intersection logic
    - If lane stale => STOP (prevents "runaway")
    """

    def __init__(self):
        super().__init__("lane_follow_imu_hold")

        # topics
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("user_cmd_topic", "/lane_follow/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        # speed
        self.declare_parameter("base_speed", 70.0)
        self.declare_parameter("min_speed", 0.0)
        self.declare_parameter("max_speed", 200.0)

        # staleness / safety
        self.declare_parameter("lane_stale_s", 0.25)
        self.declare_parameter("imu_stale_s", 0.60)
        self.declare_parameter("stop_if_lane_stale", True)

        # IMU heading hold controller
        # target_yaw = yaw_ref + (lane_offset_rad)
        self.declare_parameter("k_heading_p", 260.0)      # cmd/rad
        self.declare_parameter("k_heading_d", 55.0)       # cmd/(rad/s)
        self.declare_parameter("max_steer", 220.0)        # cmd units

        # vision -> heading offset
        # dx_norm (-1 left, +1 right) -> offset_rad
        # sign: if lane is to the left (dx<0) we need turn left => offset negative (keeps within lane)
        self.declare_parameter("dx_to_yaw_rad", 0.55)     # rad per full-scale dx_norm
        self.declare_parameter("dx_ema_alpha", 0.35)      # smoothing
        self.declare_parameter("max_dx_jump", 0.55)       # reject sudden jumps

        # motor output shaping
        self.declare_parameter("slew_per_sec", 350.0)     # limit motor command rate
        self.declare_parameter("right_bias", 0.0)         # if right motor weaker, add +bias

        # forward deadman (optional)
        self.declare_parameter("forward_deadman_s", 0.0)

        self.lane_info_topic = str(self.get_parameter("lane_info_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.user_cmd_topic = str(self.get_parameter("user_cmd_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)

        # QoS
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_motor = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos_fast)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.sub_user = self.create_subscription(String, self.user_cmd_topic, self.cb_user, qos_fast)

        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        # state
        self.latest_lane: Optional[LaneSample] = None
        self.latest_imu: Optional[ImuSample] = None

        self._yaw = 0.0
        self._last_imu_t = None

        self.running = False
        self._last_forward_cmd_t = 0.0

        # heading reference captured on "forward"
        self.yaw_ref = 0.0

        # filtered lane dx
        self.dx_filt = 0.0
        self.dx_have = False

        # motor slew state
        self._cmd_l = 0.0
        self._cmd_r = 0.0
        self._last_cmd_time = time.time()

        self.timer = self.create_timer(0.05, self.step)  # 20 Hz

        self.get_logger().info("lane_follow_imu_hold READY")
        self.get_logger().info(f"Sub lane: {self.lane_info_topic}")
        self.get_logger().info(f"Sub imu : {self.imu_topic}")
        self.get_logger().info(f"Sub cmd : {self.user_cmd_topic}")
        self.get_logger().info(f"Pub motor: {self.motor_cmd_topic}")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def cb_user(self, msg: String):
        s = (msg.data or "").strip().lower()
        if s == "forward":
            self.running = True
            self._last_forward_cmd_t = time.time()
            if self.latest_imu is not None:
                self.yaw_ref = self.latest_imu.yaw
            return

        if s == "stop":
            self.running = False
            self.publish_motor(0.0, 0.0)
            return

        # left/right ignored in this simplest mode

    def cb_lane(self, msg: String):
        now = time.time()
        try:
            d = json.loads(msg.data)
        except Exception:
            return

        # prefer "selected", else lane1
        selected = d.get("selected", None) or d.get("lane1", None)

        dx = None
        if isinstance(selected, dict):
            # lane_centerline_node: bottom_intersect_x_norm is normalized dx to center
            v = selected.get("bottom_intersect_x_norm", None)
            if v is not None:
                dx = float(v)

        self.latest_lane = LaneSample(t=now, dx_norm=dx)

        # filter dx (reject insane jumps)
        if dx is None:
            self.dx_have = False
            return

        dx = clamp(dx, -1.0, 1.0)

        if not self.dx_have:
            self.dx_filt = dx
            self.dx_have = True
            return

        if abs(dx - self.dx_filt) > float(self.get_parameter("max_dx_jump").value):
            # ignore sudden glitch frame
            return

        a = float(self.get_parameter("dx_ema_alpha").value)
        self.dx_filt = (1.0 - a) * self.dx_filt + a * dx
        self.dx_have = True

    def cb_imu(self, msg: Imu):
        now = time.time()
        wz = -float(msg.angular_velocity.z)

        if self._last_imu_t is None:
            self._last_imu_t = now
        dt = max(0.0, min(0.2, now - self._last_imu_t))
        self._last_imu_t = now

        self._yaw = wrap_pi(self._yaw + wz * dt)
        self.latest_imu = ImuSample(t=now, yaw=self._yaw, wz=wz)

    # -----------------------------
    # Main loop
    # -----------------------------
    def step(self):
        now = time.time()

        # must have fresh IMU
        imu_ok = self.latest_imu is not None and (now - self.latest_imu.t) <= float(self.get_parameter("imu_stale_s").value)
        if not imu_ok:
            self.running = False
            self.publish_motor(0.0, 0.0)
            return

        # deadman (optional): forward single-shot doesn't run forever
        dm = float(self.get_parameter("forward_deadman_s").value)
        if dm > 0.0 and self.running:
            if self._last_forward_cmd_t > 0.0 and (now - self._last_forward_cmd_t) > dm:
                self.running = False
                self.publish_motor(0.0, 0.0)
                return

        if not self.running:
            self.publish_motor(0.0, 0.0)
            return

        # lane freshness
        lane_ok = self.latest_lane is not None and (now - self.latest_lane.t) <= float(self.get_parameter("lane_stale_s").value)
        if not lane_ok or not self.dx_have:
            if bool(self.get_parameter("stop_if_lane_stale").value):
                self.publish_motor(0.0, 0.0)
                return
            # else: optionally keep going with zero trim (not recommended)
            # self.dx_filt = 0.0

        base = clamp(
            float(self.get_parameter("base_speed").value),
            float(self.get_parameter("min_speed").value),
            float(self.get_parameter("max_speed").value),
        )

        # desired heading = yaw_ref + lane_offset
        dx_to_yaw = float(self.get_parameter("dx_to_yaw_rad").value)
        lane_offset = -clamp(self.dx_filt, -1.0, 1.0) * dx_to_yaw
        yaw_target = wrap_pi(self.yaw_ref + lane_offset)

        imu = self.latest_imu
        assert imu is not None

        # heading error
        err = wrap_pi(yaw_target - imu.yaw)

        kp = float(self.get_parameter("k_heading_p").value)
        kd = float(self.get_parameter("k_heading_d").value)

        # PD on heading (D uses gyro wz)
        steer = kp * err - kd * imu.wz
        steer = clamp(steer, -float(self.get_parameter("max_steer").value), float(self.get_parameter("max_steer").value))

        # motor mix
        l = base - steer
        r = base + steer

        # right bias
        rb = float(self.get_parameter("right_bias").value)
        if r >= 0.0:
            r += rb
        else:
            r -= rb

        self.publish_motor(l, r)

    # -----------------------------
    # Slew limited motor publish
    # -----------------------------
    def publish_motor(self, l: float, r: float):
        now = time.time()
        dt = max(1e-3, now - self._last_cmd_time)
        self._last_cmd_time = now

        slew = float(self.get_parameter("slew_per_sec").value)
        max_step = slew * dt

        l = clamp(l, -1000.0, 1000.0)
        r = clamp(r, -1000.0, 1000.0)

        dl = clamp(l - self._cmd_l, -max_step, max_step)
        dr = clamp(r - self._cmd_r, -max_step, max_step)

        self._cmd_l += dl
        self._cmd_r += dr

        msg = Int16MultiArray()
        msg.data = [int(round(self._cmd_l)), int(round(self._cmd_r))]
        self.pub_motor.publish(msg)


def main():
    rclpy.init()
    node = LaneFollowImuHold()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()