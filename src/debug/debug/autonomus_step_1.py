#!/usr/bin/env python3
"""
Autonomous Step 1: Rotate to lane direction based on lane_info angle.

Davranış:
  - Komut bekler: "left" / "right".
  - Lane_info'dan selected angle_to_red_signed_deg alır.
  - Büyük açıda yüksek PWM ile döner, yaklaştıkça minimum PWM ile düzeltir.
  - Açı toleransı içindeyse durur.
"""

import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int16MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AutonomousStep1(Node):
    def __init__(self):
        super().__init__("autonomous_step_1")

        # Topics
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("turn_cmd_topic", "/lane_turn/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        # Timing
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("lane_stale_s", 1.0)

        # Lane confidence
        self.declare_parameter("conf_enter", 0.55)

        # Turn control
        self.declare_parameter("angle_tol_deg", 4.0)
        self.declare_parameter("angle_high_deg", 12.0)
        self.declare_parameter("turn_pwm_high", 320.0)
        self.declare_parameter("turn_pwm_min", 180.0)
        self.declare_parameter("turn_pwm_low", 240.0)

        # Motor
        self.declare_parameter("slew_per_sec", 500.0)

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

        self.lane_info_topic = str(self.get_parameter("lane_info_topic").value)
        self.turn_cmd_topic = str(self.get_parameter("turn_cmd_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)

        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos_fast)
        self.sub_cmd = self.create_subscription(String, self.turn_cmd_topic, self.cb_cmd, qos_fast)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        self.state = "IDLE"  # IDLE -> TURNING -> DONE
        self.turn_dir: Optional[str] = None  # "left" or "right"

        self.lane_data = None
        self.lane_last_t = 0.0

        self._last_cmd_l = 0.0
        self._last_cmd_r = 0.0
        self._last_cmd_time = time.time()

        hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, hz), self.step)

        self.get_logger().info(
            f"[READY] step1: lane_topic={self.lane_info_topic} cmd_topic={self.turn_cmd_topic}"
        )

    def cb_lane(self, msg: String):
        self.lane_last_t = time.time()
        try:
            self.lane_data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"Lane JSON parse error: {e}")

    def cb_cmd(self, msg: String):
        s = (msg.data or "").strip().lower()
        if s in ("left", "right"):
            self.turn_dir = s
            self.state = "TURNING"
            self.get_logger().info(f"[CMD] turn_dir={self.turn_dir}")
        elif s in ("stop", "cancel"):
            self.turn_dir = None
            self.state = "IDLE"
            self.publish_motor(0.0, 0.0)
            self.get_logger().info("[CMD] stop")

    def _lane_valid(self) -> bool:
        lane_stale = float(self.get_parameter("lane_stale_s").value)
        return (time.time() - self.lane_last_t) < lane_stale

    def _get_selected(self) -> Optional[dict]:
        if not isinstance(self.lane_data, dict):
            return None
        sel = self.lane_data.get("selected")
        if isinstance(sel, dict):
            return sel
        lane1 = self.lane_data.get("lane1")
        if isinstance(lane1, dict):
            return lane1
        return None

    def _get_angle_deg(self) -> Optional[float]:
        if not self._lane_valid():
            return None
        sel = self._get_selected()
        if sel is None:
            return None
        conf_enter = float(self.get_parameter("conf_enter").value)
        if float(sel.get("confidence", 0.0)) < conf_enter:
            return None
        a = sel.get("angle_to_red_signed_deg", None)
        if a is None:
            return None
        try:
            return float(a)
        except Exception:
            return None

    def step(self):
        if self.state == "IDLE":
            return

        if self.state == "TURNING":
            ang = self._get_angle_deg()
            if ang is None:
                # Lane yoksa dur ve bekle
                self.publish_motor(0.0, 0.0)
                return

            ang_abs = abs(ang)
            tol = float(self.get_parameter("angle_tol_deg").value)
            high_deg = float(self.get_parameter("angle_high_deg").value)

            if ang_abs <= tol:
                self.publish_motor(0.0, 0.0)
                self.state = "DONE"
                self.get_logger().info("[DONE] lane aligned")
                return

            # PWM selection
            pwm_high = float(self.get_parameter("turn_pwm_high").value)
            pwm_low = float(self.get_parameter("turn_pwm_low").value)
            pwm_min = float(self.get_parameter("turn_pwm_min").value)

            if ang_abs >= high_deg:
                pwm = pwm_high
            else:
                # scale between low and min as it approaches target
                t = clamp(ang_abs / max(1e-3, high_deg), 0.0, 1.0)
                pwm = pwm_min + (pwm_low - pwm_min) * t
                pwm = max(pwm, pwm_min)

            # Direction command (in-place turn)
            if self.turn_dir == "left":
                cmd_l, cmd_r = -pwm, pwm
            elif self.turn_dir == "right":
                cmd_l, cmd_r = pwm, -pwm
            else:
                cmd_l, cmd_r = 0.0, 0.0

            self.publish_motor(cmd_l, cmd_r)

        elif self.state == "DONE":
            # keep stopped until new command
            self.publish_motor(0.0, 0.0)

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
    node = AutonomousStep1()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()