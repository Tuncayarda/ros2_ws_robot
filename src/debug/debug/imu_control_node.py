#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def parse_cmd(s: str) -> Tuple[str, Optional[float]]:
    """
    Accepts:
      - "stop"
      - "forward 2"      (seconds)
      - "right 25"       (degrees)
      - "left 25"        (degrees)
    """
    s = (s or "").strip().lower()
    if not s:
        return ("", None)

    parts = s.split()
    if parts[0] == "stop":
        return ("stop", None)

    if parts[0] in ("forward", "left", "right"):
        if len(parts) < 2:
            return (parts[0], None)
        try:
            val = float(parts[1])
        except Exception:
            return (parts[0], None)
        return (parts[0], val)

    return ("", None)


@dataclass
class ImuState:
    t: float
    yaw: float     # integrated yaw (rad)
    wz: float      # yaw rate (rad/s)


class ImuOnlyMotionNode(Node):
    def __init__(self):
        super().__init__("imu_only_motion_node")

        # Topics
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("cmd_topic", "/imu_motion/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")
        self.declare_parameter("debug_topic", "/imu_motion/debug")

        # IMU sign
        self.declare_parameter("wz_sign", 1.0)

        # Freshness
        self.declare_parameter("imu_stale_s", 0.50)

        # Motor limits + slew
        self.declare_parameter("motor_limit", 1000.0)
        self.declare_parameter("slew_per_sec", 450.0)

        # Forward (heading hold)
        self.declare_parameter("fwd_speed", 180.0)
        self.declare_parameter("fwd_kp", 260.0)
        self.declare_parameter("fwd_kd", 60.0)
        self.declare_parameter("fwd_max_steer", 240.0)
        self.declare_parameter("fwd_right_bias", 0.0)

        # Turn tuning (in-place)
        self.declare_parameter("turn_base", 220.0)
        self.declare_parameter("turn_kp", 420.0)
        self.declare_parameter("turn_kd", 70.0)
        self.declare_parameter("turn_max", 520.0)

        # Turn stop logic (improved)
        self.declare_parameter("turn_stop_err_deg", 3.5)        # error threshold
        self.declare_parameter("turn_stop_wz_deg_s", 20.0)      # must be slow enough too
        self.declare_parameter("turn_stop_hold_s", 0.10)        # must hold thresholds for this time
        self.declare_parameter("turn_timeout_s", 3.5)

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
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.sub_cmd = self.create_subscription(String, self.cmd_topic, self.cb_cmd, qos_fast)

        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)
        self.pub_debug = self.create_publisher(String, self.debug_topic, qos_fast)

        # IMU integrated state (global)
        self.imu: Optional[ImuState] = None
        self._yaw = 0.0
        self._last_imu_t: Optional[float] = None

        # Action state
        self.mode = "IDLE"  # IDLE | FWD | TURN
        self.yaw_ref = 0.0
        self.action_end_t = 0.0

        # Turn-specific persisted info (for debugging / error)
        self.turn_target_yaw = 0.0
        self.turn_start_t = 0.0
        self.turn_dir = "none"          # left/right
        self.turn_cmd_deg = 0.0         # requested degrees
        self.turn_yaw0 = 0.0            # yaw at turn start
        self._turn_stop_hold_since = None

        # Motor slew state
        self._cmd_l = 0.0
        self._cmd_r = 0.0
        self._last_cmd_time = time.time()

        self.timer = self.create_timer(0.02, self.step)  # 50 Hz

        self.get_logger().info("imu_only_motion_node READY")
        self.get_logger().info(f"Sub imu : {self.imu_topic}")
        self.get_logger().info(f"Sub cmd : {self.cmd_topic}")
        self.get_logger().info(f"Pub mot : {self.motor_cmd_topic}")
        self.get_logger().info(f"Pub dbg : {self.debug_topic}")

    # ----------------------------
    # Callbacks
    # ----------------------------
    def cb_imu(self, msg: Imu):
        now = time.time()
        wz_sign = float(self.get_parameter("wz_sign").value)
        wz = wz_sign * float(msg.angular_velocity.z)

        if self._last_imu_t is None:
            self._last_imu_t = now
        dt = clamp(now - self._last_imu_t, 0.0, 0.2)
        self._last_imu_t = now

        self._yaw = wrap_pi(self._yaw + wz * dt)
        self.imu = ImuState(t=now, yaw=self._yaw, wz=wz)

    def cb_cmd(self, msg: String):
        cmd, val = parse_cmd(msg.data)

        if cmd == "stop":
            self.mode = "IDLE"
            self._turn_stop_hold_since = None
            self.publish_motor(0.0, 0.0, force=True)
            self.debug("STOP")
            return

        if self.imu is None:
            self.get_logger().warning("IMU not ready; ignoring cmd")
            return

        if cmd == "forward":
            if val is None or val <= 0.0:
                self.get_logger().warning("forward requires seconds: 'forward 2'")
                return
            self.mode = "FWD"
            self.yaw_ref = self.imu.yaw
            self.action_end_t = time.time() + float(val)
            self.debug(f"FWD start {val:.2f}s yaw_ref={rad2deg(self.yaw_ref):.2f}deg")
            return

        if cmd in ("left", "right"):
            if val is None or val <= 0.0:
                self.get_logger().warning("turn requires degrees: 'right 25'")
                return

            deg = float(val)
            delta = deg2rad(deg)
            yaw0 = self.imu.yaw

            # left => +delta, right => -delta (given your wz sign convention)
            target = wrap_pi(yaw0 + delta) if cmd == "left" else wrap_pi(yaw0 - delta)

            self.mode = "TURN"
            self.turn_start_t = time.time()
            self.turn_dir = cmd
            self.turn_cmd_deg = deg
            self.turn_yaw0 = yaw0
            self.turn_target_yaw = target
            self._turn_stop_hold_since = None

            self.debug(
                f"TURN start dir={cmd} cmd_deg={deg:.1f} yaw0={rad2deg(yaw0):.2f} "
                f"target={rad2deg(target):.2f}"
            )
            return

    # ----------------------------
    # Main loop
    # ----------------------------
    def step(self):
        now = time.time()

        imu_stale = float(self.get_parameter("imu_stale_s").value)
        if self.imu is None or (now - self.imu.t) > imu_stale:
            self.mode = "IDLE"
            self._turn_stop_hold_since = None
            self.publish_motor(0.0, 0.0)
            return

        if self.mode == "IDLE":
            self.publish_motor(0.0, 0.0)
            return

        if self.mode == "FWD":
            if now >= self.action_end_t:
                self.mode = "IDLE"
                self.publish_motor(0.0, 0.0, force=True)
                self.debug("FWD done")
                return
            self.step_forward_heading_hold()
            return

        if self.mode == "TURN":
            self.step_turn_in_place()
            return

        self.mode = "IDLE"
        self.publish_motor(0.0, 0.0)

    # ----------------------------
    # Controllers
    # ----------------------------
    def step_forward_heading_hold(self):
        imu = self.imu
        assert imu is not None

        base = float(self.get_parameter("fwd_speed").value)
        kp = float(self.get_parameter("fwd_kp").value)
        kd = float(self.get_parameter("fwd_kd").value)
        max_steer = float(self.get_parameter("fwd_max_steer").value)

        err = wrap_pi(self.yaw_ref - imu.yaw)
        steer = kp * err - kd * imu.wz
        steer = clamp(steer, -max_steer, max_steer)

        l = base - steer
        r = base + steer

        rb = float(self.get_parameter("fwd_right_bias").value)
        if r >= 0.0:
            r += rb
        else:
            r -= rb

        self.publish_motor(l, r)

    def step_turn_in_place(self):
        imu = self.imu
        assert imu is not None

        now = time.time()
        timeout = float(self.get_parameter("turn_timeout_s").value)
        if (now - self.turn_start_t) > timeout:
            self.publish_motor(0.0, 0.0, force=True)
            self.finish_turn(reason="timeout")
            return

        err = wrap_pi(self.turn_target_yaw - imu.yaw)   # rad
        err_deg = abs(rad2deg(err))
        wz_deg_s = abs(rad2deg(imu.wz))

        stop_err = float(self.get_parameter("turn_stop_err_deg").value)
        stop_wz = float(self.get_parameter("turn_stop_wz_deg_s").value)
        hold_s = float(self.get_parameter("turn_stop_hold_s").value)

        # require BOTH: small error and low angular velocity, held for hold_s
        if (err_deg <= stop_err) and (wz_deg_s <= stop_wz):
            if self._turn_stop_hold_since is None:
                self._turn_stop_hold_since = now
            if (now - self._turn_stop_hold_since) >= hold_s:
                self.publish_motor(0.0, 0.0, force=True)
                self.finish_turn(reason="threshold_hold")
                return
        else:
            self._turn_stop_hold_since = None

        kp = float(self.get_parameter("turn_kp").value)
        kd = float(self.get_parameter("turn_kd").value)
        base = float(self.get_parameter("turn_base").value)
        u_max = float(self.get_parameter("turn_max").value)

        u_pd = kp * err - kd * imu.wz
        mag = clamp(abs(u_pd) + base, 0.0, u_max)

        # err>0 => yaw should increase => turn left
        sign = 1.0 if err >= 0.0 else -1.0

        l = -sign * mag
        r = +sign * mag

        self.publish_motor(l, r)

    def finish_turn(self, reason: str):
        imu = self.imu
        assert imu is not None

        # compute final error vs target
        err_final = wrap_pi(self.turn_target_yaw - imu.yaw)
        err_final_deg = rad2deg(err_final)

        # also compute actual delta moved vs requested
        delta_actual = wrap_pi(imu.yaw - self.turn_yaw0)
        delta_actual_deg = rad2deg(delta_actual)
        delta_cmd = self.turn_cmd_deg if self.turn_dir == "left" else -self.turn_cmd_deg
        delta_err_deg = delta_cmd - delta_actual_deg

        msg = (
            f"TURN DONE dir={self.turn_dir} reason={reason} "
            f"cmd_deg={self.turn_cmd_deg:.1f} "
            f"yaw0={rad2deg(self.turn_yaw0):.2f} "
            f"target={rad2deg(self.turn_target_yaw):.2f} "
            f"yaw={rad2deg(imu.yaw):.2f} "
            f"err_to_target_deg={err_final_deg:.2f} "
            f"delta_actual_deg={delta_actual_deg:.2f} "
            f"delta_err_deg={delta_err_deg:.2f} "
            f"elapsed={time.time()-self.turn_start_t:.2f}s"
        )
        self.get_logger().info(msg)
        self.debug(msg)

        self.mode = "IDLE"
        self._turn_stop_hold_since = None

    def debug(self, s: str):
        m = String()
        m.data = s
        self.pub_debug.publish(m)

    # ----------------------------
    # Motor publish with slew
    # ----------------------------
    def publish_motor(self, l: float, r: float, force: bool = False):
        now = time.time()
        dt = max(1e-3, now - self._last_cmd_time)
        self._last_cmd_time = now

        limit = float(self.get_parameter("motor_limit").value)
        slew = float(self.get_parameter("slew_per_sec").value)
        max_step = slew * dt

        l = clamp(l, -limit, limit)
        r = clamp(r, -limit, limit)

        if force:
            self._cmd_l = l
            self._cmd_r = r
        else:
            dl = clamp(l - self._cmd_l, -max_step, max_step)
            dr = clamp(r - self._cmd_r, -max_step, max_step)
            self._cmd_l += dl
            self._cmd_r += dr

        msg = Int16MultiArray()
        msg.data = [int(round(self._cmd_l)), int(round(self._cmd_r))]
        self.pub_motor.publish(msg)


def main():
    rclpy.init()
    node = ImuOnlyMotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()