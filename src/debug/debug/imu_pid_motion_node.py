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
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def parse_cmd(s: str) -> Tuple[str, Optional[float]]:
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
    yaw: float
    wz: float  # rad/s


class ImuPidMotionNode(Node):
    """
    IMU-only motion executor with:
      - Forward: PD heading hold
      - Turn: PID (deg domain) + per-wheel minimum PWM enforcement

    Topics:
      Sub: /imu, /imu_motion/cmd
      Pub: /motor_cmd, /imu_motion/debug
    """

    def __init__(self):
        super().__init__("imu_pid_motion_node")

        # Topics
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("cmd_topic", "/imu_motion/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")
        self.declare_parameter("debug_topic", "/imu_motion/debug")

        # IMU signs
        self.declare_parameter("yaw_sign", 1.0)
        self.declare_parameter("wz_sign", 1.0)

        # If turn direction inverted:
        self.declare_parameter("turn_motor_sign", 1.0)

        # Freshness
        self.declare_parameter("imu_stale_s", 0.50)

        # Motor limits + slew
        self.declare_parameter("motor_limit", 1000.0)
        self.declare_parameter("slew_per_sec", 900.0)

        # Forward heading hold (PD)
        self.declare_parameter("fwd_speed", 180.0)
        self.declare_parameter("fwd_kp", 260.0)
        self.declare_parameter("fwd_kd", 60.0)
        self.declare_parameter("fwd_max_steer", 260.0)
        self.declare_parameter("fwd_right_bias", 0.0)

        # TURN PID (deg domain)
        self.declare_parameter("turn_kp", 7.5)         # PWM/deg
        self.declare_parameter("turn_ki", 0.35)        # PWM/(deg*s)  <-- ENABLED
        self.declare_parameter("turn_kd", 1.6)         # PWM/(deg/s)

        # Anti-windup
        self.declare_parameter("turn_i_clamp_pwm", 260.0)   # clamp I contribution (PWM units)
        self.declare_parameter("turn_i_zone_deg", 35.0)     # integrate only when |err| <= this
        self.declare_parameter("turn_i_leak", 0.06)         # 0..1 per second, bleed I down (prevents drift)

        # Turn style
        self.declare_parameter("turn_mode", "inplace")  # "inplace" or "arc"

        # Per-wheel minimum PWM during TURN (your hard requirement)
        self.declare_parameter("turn_wheel_min_pwm", 240.0)

        # Overall bounds for turn effort (magnitude of differential)
        self.declare_parameter("turn_mag_max_pwm", 750.0)

        # Near-target shaping (optional)
        # NOTE: wheel_min is still enforced, so this only caps the "extra" effort.
        self.declare_parameter("fine_band_deg", 8.0)
        self.declare_parameter("fine_mag_max_pwm", 520.0)

        # Stop condition (hold)
        self.declare_parameter("turn_tol_deg", 2.0)
        self.declare_parameter("turn_stop_wz_deg_s", 16.0)
        self.declare_parameter("turn_settle_s", 0.22)

        # Timeout
        self.declare_parameter("turn_timeout_s", 6.0)

        # Control frequency
        self.declare_parameter("control_hz", 60.0)

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

        # IMU integrated state
        self.imu: Optional[ImuState] = None
        self._yaw = 0.0
        self._last_imu_t: Optional[float] = None

        # Mode
        self.mode = "IDLE"  # IDLE | FWD | TURN
        self.yaw_ref = 0.0
        self.fwd_end_t = 0.0

        # TURN state
        self.turn_target_yaw = 0.0
        self.turn_start_t = 0.0
        self.turn_dir = "none"
        self.turn_cmd_deg = 0.0
        self.turn_yaw0 = 0.0
        self._stop_hold_since: Optional[float] = None

        # PID internals
        self._i_term_deg_s = 0.0
        self._prev_err_deg: Optional[float] = None
        self._prev_t: Optional[float] = None

        # motor slew state
        self._cmd_l = 0.0
        self._cmd_r = 0.0
        self._last_cmd_time = time.time()

        hz = float(self.get_parameter("control_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, hz), self.step)

        self.get_logger().info("imu_pid_motion_node READY")

    # ----------------------------
    # IMU
    # ----------------------------
    def cb_imu(self, msg: Imu):
        now = time.time()
        yaw_sign = float(self.get_parameter("yaw_sign").value)
        wz_sign = float(self.get_parameter("wz_sign").value)

        wz_raw = float(msg.angular_velocity.z)
        wz = wz_sign * wz_raw

        if self._last_imu_t is None:
            self._last_imu_t = now
        dt = clamp(now - self._last_imu_t, 0.0, 0.2)
        self._last_imu_t = now

        self._yaw = wrap_pi(self._yaw + (yaw_sign * wz_raw) * dt)
        self.imu = ImuState(t=now, yaw=self._yaw, wz=wz)

    # ----------------------------
    # CMD
    # ----------------------------
    def cb_cmd(self, msg: String):
        cmd, val = parse_cmd(msg.data)

        if cmd == "stop":
            self.set_idle("STOP")
            return

        if self.imu is None:
            self.warn_dbg("IMU not ready; ignoring cmd")
            return

        if cmd == "forward":
            if val is None or val <= 0.0:
                self.warn_dbg("forward requires seconds: 'forward 2'")
                return
            self.mode = "FWD"
            self.yaw_ref = self.imu.yaw
            self.fwd_end_t = time.time() + float(val)
            self.dbg(f"FWD start {val:.2f}s yaw_ref={rad2deg(self.yaw_ref):.2f}")
            return

        if cmd in ("left", "right"):
            if val is None or val <= 0.0:
                self.warn_dbg("turn requires degrees: 'left 90'")
                return

            deg = float(val)
            yaw0 = self.imu.yaw
            delta = deg2rad(deg)
            target = wrap_pi(yaw0 + delta) if cmd == "left" else wrap_pi(yaw0 - delta)

            self.mode = "TURN"
            self.turn_start_t = time.time()
            self.turn_dir = cmd
            self.turn_cmd_deg = deg
            self.turn_yaw0 = yaw0
            self.turn_target_yaw = target

            # reset PID memory
            self._i_term_deg_s = 0.0
            self._prev_err_deg = None
            self._prev_t = None
            self._stop_hold_since = None

            self.dbg(f"TURN start dir={cmd} deg={deg:.1f} yaw0={rad2deg(yaw0):.2f} target={rad2deg(target):.2f}")
            return

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    def step(self):
        now = time.time()
        imu_stale = float(self.get_parameter("imu_stale_s").value)

        if self.imu is None or (now - self.imu.t) > imu_stale:
            self.set_idle("IMU STALE")
            return

        if self.mode == "IDLE":
            self.publish_motor(0.0, 0.0)
            return

        if self.mode == "FWD":
            if now >= self.fwd_end_t:
                self.set_idle("FWD done")
                return
            self.step_forward()
            return

        if self.mode == "TURN":
            self.step_turn_pid()
            return

        self.set_idle("UNKNOWN MODE")

    # ----------------------------
    # FORWARD PD
    # ----------------------------
    def step_forward(self):
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
        r = r + rb if r >= 0.0 else r - rb

        self.publish_motor(l, r)

    # ----------------------------
    # TURN PID (deg domain) + per-wheel minimum enforcement
    # ----------------------------
    def step_turn_pid(self):
        imu = self.imu
        assert imu is not None

        now = time.time()
        if (now - self.turn_start_t) > float(self.get_parameter("turn_timeout_s").value):
            self.finish_turn("timeout")
            return

        err_deg = rad2deg(wrap_pi(self.turn_target_yaw - imu.yaw))
        wz_deg_s = rad2deg(imu.wz)

        # stop condition (hold)
        tol = float(self.get_parameter("turn_tol_deg").value)
        stop_wz = float(self.get_parameter("turn_stop_wz_deg_s").value)
        settle = float(self.get_parameter("turn_settle_s").value)
        if (abs(err_deg) <= tol) and (abs(wz_deg_s) <= stop_wz):
            if self._stop_hold_since is None:
                self._stop_hold_since = now
            if (now - self._stop_hold_since) >= settle:
                self.finish_turn("settled")
                return
        else:
            self._stop_hold_since = None

        # dt
        if self._prev_t is None:
            dt = 1.0 / max(1.0, float(self.get_parameter("control_hz").value))
        else:
            dt = clamp(now - self._prev_t, 1e-3, 0.1)
        self._prev_t = now

        # PID params
        kp = float(self.get_parameter("turn_kp").value)
        ki = float(self.get_parameter("turn_ki").value)
        kd = float(self.get_parameter("turn_kd").value)

        # derivative of error (deg/s)
        if self._prev_err_deg is None:
            derr = 0.0
        else:
            derr = (err_deg - self._prev_err_deg) / dt
        self._prev_err_deg = err_deg

        # Integral: only near-ish target (I-zone) + leak
        i_zone = float(self.get_parameter("turn_i_zone_deg").value)
        leak = float(self.get_parameter("turn_i_leak").value)  # per second

        # leak (exponential-ish)
        self._i_term_deg_s *= max(0.0, 1.0 - leak * dt)

        if ki > 0.0 and abs(err_deg) <= i_zone:
            self._i_term_deg_s += err_deg * dt

        # anti-windup clamp by limiting I contribution in PWM units
        i_clamp_pwm = float(self.get_parameter("turn_i_clamp_pwm").value)
        if ki > 0.0:
            self._i_term_deg_s = clamp(self._i_term_deg_s, -i_clamp_pwm / ki, i_clamp_pwm / ki)
        else:
            self._i_term_deg_s = 0.0

        # PID output (PWM-ish)
        u = kp * err_deg + ki * self._i_term_deg_s + kd * derr
        mag = abs(u)

        # magnitude caps
        mag_max = float(self.get_parameter("turn_mag_max_pwm").value)
        fine_band = float(self.get_parameter("fine_band_deg").value)
        fine_mag_max = float(self.get_parameter("fine_mag_max_pwm").value)

        if abs(err_deg) <= fine_band:
            mag = clamp(mag, 0.0, fine_mag_max)
        else:
            mag = clamp(mag, 0.0, mag_max)

        # direction
        sign = 1.0 if err_deg >= 0.0 else -1.0
        sign *= float(self.get_parameter("turn_motor_sign").value)

        turn_mode = str(self.get_parameter("turn_mode").value).strip().lower()
        wheel_min = float(self.get_parameter("turn_wheel_min_pwm").value)

        if turn_mode == "arc":
            base = wheel_min
            steer = clamp(mag, 0.0, float(self.get_parameter("motor_limit").value))
            if sign > 0:   # left
                l = base
                r = base + steer
            else:
                l = base + steer
                r = base
        else:
            # inplace
            l = -sign * mag
            r = +sign * mag

            # enforce per-wheel minimum ALWAYS while turning
            if abs(l) < wheel_min:
                l = math.copysign(wheel_min, l if l != 0.0 else -sign)
            if abs(r) < wheel_min:
                r = math.copysign(wheel_min, r if r != 0.0 else sign)

        self.publish_motor(l, r)

    # ----------------------------
    # FINISH / IDLE
    # ----------------------------
    def set_idle(self, reason: str):
        self.mode = "IDLE"
        self._stop_hold_since = None
        self.publish_motor(0.0, 0.0, force=True)
        self.dbg(reason)

    def finish_turn(self, reason: str):
        imu = self.imu
        assert imu is not None

        self.publish_motor(0.0, 0.0, force=True)

        err_final_deg = rad2deg(wrap_pi(self.turn_target_yaw - imu.yaw))
        delta_actual_deg = rad2deg(wrap_pi(imu.yaw - self.turn_yaw0))
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
        self.dbg(msg)

        self.mode = "IDLE"
        self._stop_hold_since = None

    def dbg(self, s: str):
        m = String()
        m.data = s
        self.pub_debug.publish(m)

    def warn_dbg(self, s: str):
        self.get_logger().warning(s)
        self.dbg("WARN: " + s)

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
    node = ImuPidMotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()