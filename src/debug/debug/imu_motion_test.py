#!/usr/bin/env python3
"""
IMU-only straight drive test.
Robot düz gitmeye çalışır, yaw sabit tutulur.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from std_msgs.msg import Int16MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ImuState:
    t: float
    yaw: float
    wz: float


class ImuStraightNode(Node):
    def __init__(self):
        super().__init__("imu_straight_test")

        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("imu_stale_s", 0.5)

        self.declare_parameter("start_speed", 40.0)
        self.declare_parameter("cruise_speed", 140.0)
        self.declare_parameter("ramp_time_s", 2.0)

        self.declare_parameter("heading_kp", 260.0)
        self.declare_parameter("heading_kd", 10.0)
        self.declare_parameter("heading_ki", 0.0)
        self.declare_parameter("heading_steer_sign", -1.0)
        self.declare_parameter("wz_sign", -1.0)
        self.declare_parameter("max_heading_trim", 220.0)

        self.declare_parameter("slew_per_sec", 700.0)

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

        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)

        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        self.imu: Optional[ImuState] = None
        self._yaw_ref = 0.0
        self._yaw_err_i = 0.0
        self._bias = 0.0
        self._start_t = time.time()
        self._last_step_t = time.time()

        self._last_cmd_l = 0.0
        self._last_cmd_r = 0.0
        self._last_cmd_time = time.time()

        hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, hz), self.step)

        self.get_logger().info(f"READY imu_straight_test imu={self.imu_topic}")

    def cb_imu(self, msg: Imu):
        now = time.time()
        yaw = yaw_from_quat(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        yaw = wrap_pi(yaw)
        wz = msg.angular_velocity.z * float(self.get_parameter("wz_sign").value)
        if self.imu is None:
            self._yaw_ref = yaw
            self._start_t = now
            self.get_logger().info(f"Yaw ref set: {self._yaw_ref:.3f} rad")
        self.imu = ImuState(t=now, yaw=yaw, wz=wz)

    def _heading_trim(self, dt: float) -> float:
        if self.imu is None:
            return 0.0
        kp = float(self.get_parameter("heading_kp").value)
        kd = float(self.get_parameter("heading_kd").value)
        ki = float(self.get_parameter("heading_ki").value)
        steer_sign = float(self.get_parameter("heading_steer_sign").value)

        yaw_err = wrap_pi(self._yaw_ref - self.imu.yaw)
        if ki != 0.0:
            self._yaw_err_i += yaw_err * dt
            self._yaw_err_i = clamp(self._yaw_err_i, -0.6, 0.6)
        else:
            self._yaw_err_i = 0.0

        trim = (kp * yaw_err) - (kd * self.imu.wz) + (ki * self._yaw_err_i)
        trim = clamp(trim * steer_sign,
                     -float(self.get_parameter("max_heading_trim").value),
                     float(self.get_parameter("max_heading_trim").value))
        return trim

    def step(self):
        now = time.time()
        dt = max(0.001, now - self._last_step_t)
        self._last_step_t = now

        if self.imu is None or (now - self.imu.t) > float(self.get_parameter("imu_stale_s").value):
            self.publish_motor(0.0, 0.0)
            return

        start_speed = float(self.get_parameter("start_speed").value)
        cruise_speed = float(self.get_parameter("cruise_speed").value)
        ramp_time = float(self.get_parameter("ramp_time_s").value)

        t = (now - self._start_t) / max(0.1, ramp_time)
        speed = start_speed + (cruise_speed - start_speed) * clamp(t, 0.0, 1.0)

        trim = self._heading_trim(dt)
        # auto bias compensation for asymmetric drivetrain
        # positive yaw_err => drifting right, add bias to right motor
        yaw_err = wrap_pi(self._yaw_ref - self.imu.yaw)
        self._bias += clamp(yaw_err, -0.06, 0.06) * 60.0 * dt
        self._bias = clamp(self._bias, -120.0, 120.0)

        cmd_l = speed + trim - self._bias
        cmd_r = speed - trim + self._bias

        self.publish_motor(cmd_l, cmd_r)

    def publish_motor(self, cmd_l: float, cmd_r: float):
        now = time.time()
        dt = max(0.001, now - self._last_cmd_time)
        slew_per_sec = float(self.get_parameter("slew_per_sec").value)
        max_delta = slew_per_sec * dt

        cmd_l = clamp(cmd_l, self._last_cmd_l - max_delta, self._last_cmd_l + max_delta)
        cmd_r = clamp(cmd_r, self._last_cmd_r - max_delta, self._last_cmd_r + max_delta)

        self._last_cmd_l = cmd_l
        self._last_cmd_r = cmd_r
        self._last_cmd_time = now

        cmd_l = clamp(cmd_l, -1000, 1000)
        cmd_r = clamp(cmd_r, -1000, 1000)

        msg = Int16MultiArray()
        msg.data = [int(round(cmd_l)), int(round(cmd_r))]
        self.pub_motor.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuStraightNode()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()