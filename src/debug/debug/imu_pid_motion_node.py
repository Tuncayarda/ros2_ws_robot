#!/usr/bin/env python3
"""
IMU PID rotation node.
Komut ile verilen açı ve yöne göre robotu döndürür.

Komut formatı (String):
  "left 90" veya "right 45" veya "-90" (negatif = left)
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from std_msgs.msg import String, Int16MultiArray


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


class ImuPidRotate(Node):
    def __init__(self):
        super().__init__("imu_pid_rotate")

        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("cmd_topic", "/imu_turn/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("imu_stale_s", 0.5)

        self.declare_parameter("kp", 6.0)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 18.0)
        self.declare_parameter("wz_sign", -1.0)

        self.declare_parameter("pwm_max", 380.0)
        self.declare_parameter("pwm_min", 180.0)
        self.declare_parameter("angle_tol_deg", 3.0)
        self.declare_parameter("settle_frames", 4)

        self.declare_parameter("slew_per_sec", 900.0)

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
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)

        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.sub_cmd = self.create_subscription(String, self.cmd_topic, self.cb_cmd, qos_fast)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        self.imu: Optional[ImuState] = None
        self.target_yaw: Optional[float] = None
        self.target_yaw_global: Optional[float] = None
        self.running = False

        self._err_i = 0.0
        self._last_err = None
        self._settle = 0

        self._yaw_prev: Optional[float] = None
        self._yaw_global = 0.0

        self._last_cmd_l = 0.0
        self._last_cmd_r = 0.0
        self._last_cmd_time = time.time()

        hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, hz), self.step)

        self.get_logger().info(f"READY imu_pid_rotate cmd={self.cmd_topic}")

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
        if self._yaw_prev is None:
            self._yaw_prev = yaw
            self._yaw_global = yaw
        else:
            dy = wrap_pi(yaw - self._yaw_prev)
            self._yaw_global += dy
            self._yaw_prev = yaw
        self.imu = ImuState(t=now, yaw=yaw, wz=wz)

    def cb_cmd(self, msg: String):
        text = (msg.data or "").strip().lower()
        angle_deg, direction = self._parse_cmd(text)
        if angle_deg is None or direction is None:
            self.get_logger().warn("Invalid cmd. Use: 'left 90' or 'right 45' or '-90'")
            return
        if self.imu is None:
            self.get_logger().warn("IMU not ready")
            return
        sign = -1.0 if direction == "left" else 1.0
        target = wrap_pi(self.imu.yaw + sign * math.radians(angle_deg))
        self.target_yaw = target
        self.target_yaw_global = self._yaw_global + sign * math.radians(angle_deg)
        self.running = True
        self._err_i = 0.0
        self._last_err = None
        self._settle = 0
        self.get_logger().info(f"Turn cmd: {direction} {angle_deg:.1f}deg -> target_yaw={target:.3f}")

    def _parse_cmd(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        if not text:
            return None, None
        parts = text.split()
        if len(parts) == 1:
            try:
                deg = float(parts[0])
                return abs(deg), ("left" if deg < 0 else "right")
            except Exception:
                return None, None
        if len(parts) >= 2:
            dir_s = parts[0]
            if dir_s not in ("left", "right"):
                return None, None
            try:
                deg = abs(float(parts[1]))
            except Exception:
                return None, None
            return deg, dir_s
        return None, None

    def step(self):
        now = time.time()
        if self.imu is None or (now - self.imu.t) > float(self.get_parameter("imu_stale_s").value):
            self.publish_motor(0.0, 0.0)
            return

        if not self.running or self.target_yaw is None:
            return

        err = wrap_pi(self.target_yaw - self.imu.yaw)
        err_global = self.target_yaw_global - self._yaw_global if self.target_yaw_global is not None else err
        err_deg = abs(math.degrees(err_global))
        tol = float(self.get_parameter("angle_tol_deg").value)
        settle_frames = int(self.get_parameter("settle_frames").value)

        if err_deg <= tol:
            self._settle += 1
            if self._settle >= settle_frames:
                self.publish_motor(0.0, 0.0)
                self.running = False
                self.get_logger().info("Turn complete")
                return
        else:
            self._settle = 0

        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)

        if self._last_err is None:
            derr = 0.0
        else:
            dt = max(0.001, now - self.imu.t)
            derr = (err_global - self._last_err) / dt
        self._last_err = err_global

        if ki != 0.0:
            self._err_i += err * 0.02
            self._err_i = clamp(self._err_i, -0.6, 0.6)
        else:
            self._err_i = 0.0

        pwm = kp * abs(err_global) + ki * abs(self._err_i) - kd * abs(derr)
        pwm = clamp(pwm, float(self.get_parameter("pwm_min").value), float(self.get_parameter("pwm_max").value))

        if err_global > 0:
            cmd_l, cmd_r = pwm, -pwm
        else:
            cmd_l, cmd_r = -pwm, pwm

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
    node = ImuPidRotate()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()