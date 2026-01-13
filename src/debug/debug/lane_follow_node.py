#!/usr/bin/env python3
import time
import math
import json
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Imu


# ---------------- utils
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def lerp(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def wrap_deg(a: float) -> float:
    while a >= 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a

def lpf(prev: float, x: float, alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return prev + alpha * (x - prev)

def softsign(x: float, k: float) -> float:
    k = max(1e-6, float(k))
    return math.tanh(x / k)

def sgn(x: float) -> float:
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)

def pow_ease(x: float, gamma: float) -> float:
    """x in [0..1] -> [0..1], gamma>1 => near 0 becomes much smaller (nice approach)."""
    x = clamp(x, 0.0, 1.0)
    gamma = max(0.05, float(gamma))
    return x ** gamma


class LaneFollowNode(Node):
    """
    lane_follow_node (lane1 only)

    NEW (as requested):
      - Nonlinear approach: desired heading offset saturates with an ease curve:
          approach = max_approach_deg * (|e|^gamma)
        so when you are close (|e| small) the robot enters with a small angle,
        and when far it can be more aggressive.
      - Keep turn boost so it doesn't "ıkına ıkına" while turning.
    """

    def __init__(self):
        super().__init__("lane_follow_node")

        # topics
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("motor_topic", "/motor_cmd")

        # rates / timeouts
        self.declare_parameter("control_hz", 80.0)
        self.declare_parameter("imu_timeout_s", 0.35)
        self.declare_parameter("lane_timeout_s", 0.8)

        # speed (scaled by speed_scale)
        self.declare_parameter("base_pwm", 150.0)         # biraz daha canlı default
        self.declare_parameter("min_base_pwm", 110.0)
        self.declare_parameter("max_base_pwm", 300.0)
        self.declare_parameter("speed_scale", 0.6)

        self.declare_parameter("stop_if_lane_lost", False)

        # signs
        self.declare_parameter("yaw_sign", -1.0)
        self.declare_parameter("turn_motor_sign", 1.0)
        self.declare_parameter("lane_sign", 1.0)

        # lane selection
        self.declare_parameter("require_valid_lane", False)

        # lane signal pick
        self.declare_parameter("use_mid_term", True)
        self.declare_parameter("lane_deadband", 0.03)

        # ---------------- NONLINEAR APPROACH (MAIN KNOBS)
        self.declare_parameter("enable_nonlinear_approach", True)

        # max heading offset purely from position error (deg)
        self.declare_parameter("max_approach_deg", 18.0)   # <- "40 yerine 10-20" hedefin için
        # curvature: higher => near center much gentler (natural)
        self.declare_parameter("approach_gamma", 1.8)      # 1.4-2.4 arası güzel

        # add a small mid-term help (optional)
        self.declare_parameter("mid_mix", 0.35)            # 0..1, 0.25-0.45 iyi

        # small lane-angle correction (deg contribution)
        self.declare_parameter("use_lane_angle_term", True)
        self.declare_parameter("kp_angle_deg_per_deg", 0.22)   # eskisi 0.40 agresifti
        self.declare_parameter("max_angle_term_deg", 8.0)      # angle term clamp

        # soften raw signals (pre-shaping)
        self.declare_parameter("soft_steer", True)
        self.declare_parameter("offset_soft_k", 0.55)      # büyüt => daha yumuşak
        self.declare_parameter("angle_soft_k_deg", 35.0)

        # confidence gating
        self.declare_parameter("conf_gate", 0.30)
        self.declare_parameter("pair_boost", 1.15)

        # clamp + smoothing (offset dynamics)
        self.declare_parameter("yaw_offset_limit_deg", 22.0)
        self.declare_parameter("desired_offset_rate_deg_s", 65.0)  # çok yavaş zigzag olmasın
        self.declare_parameter("offset_lpf_alpha", 0.20)           # daha hızlı tepki

        # yaw PID -> diff pwm
        self.declare_parameter("kp", 6.5)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 1.0)
        self.declare_parameter("i_clamp_pwm", 220.0)
        self.declare_parameter("max_diff_pwm", 280.0)

        # NEW: base boost when turning (diff large)
        self.declare_parameter("enable_turn_boost", True)
        self.declare_parameter("turn_base_gain", 1.6)         # 2.0 bazen "yardırıyor", 1.4-1.8 daha kontrollü
        self.declare_parameter("turn_boost_start_pwm", 70.0)  # erken başlamasın
        self.declare_parameter("turn_boost_full_pwm", 180.0)
        self.declare_parameter("turn_boost_max_base_pwm", 360.0)

        # motor map + slew
        self.declare_parameter("slew_per_tick", 85)  # ıkınmasın diye biraz daha hızlı ramp
        self.declare_parameter("left_trim", 0.0)
        self.declare_parameter("right_trim", 0.0)
        self.declare_parameter("left_scale", 1.0)
        self.declare_parameter("right_scale", 1.0)

        # speed policy
        self.declare_parameter("slow_on_error", True)
        self.declare_parameter("slow_err_norm_at", 0.55)   # daha geç yavaşlasın (zigzag azalır)
        self.declare_parameter("slow_angle_deg_at", 22.0)
        self.declare_parameter("slow_min_ratio", 0.70)

        # debug
        self.declare_parameter("debug_hz", 5.0)

        # read
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.lane_info_topic = str(self.get_parameter("lane_info_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.imu_timeout_s = float(self.get_parameter("imu_timeout_s").value)
        self.lane_timeout_s = float(self.get_parameter("lane_timeout_s").value)

        self.base_pwm = float(self.get_parameter("base_pwm").value)
        self.min_base_pwm = float(self.get_parameter("min_base_pwm").value)
        self.max_base_pwm = float(self.get_parameter("max_base_pwm").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.stop_if_lane_lost = bool(self.get_parameter("stop_if_lane_lost").value)

        self.yaw_sign = float(self.get_parameter("yaw_sign").value)
        self.turn_motor_sign = float(self.get_parameter("turn_motor_sign").value)
        self.lane_sign = float(self.get_parameter("lane_sign").value)

        self.require_valid_lane = bool(self.get_parameter("require_valid_lane").value)

        self.use_mid_term = bool(self.get_parameter("use_mid_term").value)
        self.lane_deadband = float(self.get_parameter("lane_deadband").value)

        self.enable_nonlinear_approach = bool(self.get_parameter("enable_nonlinear_approach").value)
        self.max_approach_deg = float(self.get_parameter("max_approach_deg").value)
        self.approach_gamma = float(self.get_parameter("approach_gamma").value)
        self.mid_mix = float(self.get_parameter("mid_mix").value)

        self.use_lane_angle_term = bool(self.get_parameter("use_lane_angle_term").value)
        self.kp_ang = float(self.get_parameter("kp_angle_deg_per_deg").value)
        self.max_angle_term_deg = float(self.get_parameter("max_angle_term_deg").value)

        self.soft_steer = bool(self.get_parameter("soft_steer").value)
        self.offset_soft_k = float(self.get_parameter("offset_soft_k").value)
        self.angle_soft_k_deg = float(self.get_parameter("angle_soft_k_deg").value)

        self.conf_gate = float(self.get_parameter("conf_gate").value)
        self.pair_boost = float(self.get_parameter("pair_boost").value)

        self.yaw_offset_limit_deg = float(self.get_parameter("yaw_offset_limit_deg").value)
        self.desired_offset_rate_deg_s = float(self.get_parameter("desired_offset_rate_deg_s").value)
        self.offset_lpf_alpha = float(self.get_parameter("offset_lpf_alpha").value)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.i_clamp_pwm = float(self.get_parameter("i_clamp_pwm").value)
        self.max_diff_pwm = float(self.get_parameter("max_diff_pwm").value)

        self.enable_turn_boost = bool(self.get_parameter("enable_turn_boost").value)
        self.turn_base_gain = float(self.get_parameter("turn_base_gain").value)
        self.turn_boost_start_pwm = float(self.get_parameter("turn_boost_start_pwm").value)
        self.turn_boost_full_pwm = float(self.get_parameter("turn_boost_full_pwm").value)
        self.turn_boost_max_base_pwm = float(self.get_parameter("turn_boost_max_base_pwm").value)

        self.slew_per_tick = int(self.get_parameter("slew_per_tick").value)
        self.left_trim = float(self.get_parameter("left_trim").value)
        self.right_trim = float(self.get_parameter("right_trim").value)
        self.left_scale = float(self.get_parameter("left_scale").value)
        self.right_scale = float(self.get_parameter("right_scale").value)

        self.slow_on_error = bool(self.get_parameter("slow_on_error").value)
        self.slow_err_norm_at = float(self.get_parameter("slow_err_norm_at").value)
        self.slow_angle_deg_at = float(self.get_parameter("slow_angle_deg_at").value)
        self.slow_min_ratio = float(self.get_parameter("slow_min_ratio").value)

        self.debug_hz = float(self.get_parameter("debug_hz").value)
        self._dbg_period = (1.0 / self.debug_hz) if self.debug_hz > 0 else 0.0
        self._dbg_last = 0.0

        # ROS
        qos_imu = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=50)
        qos_lane = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_imu)
        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos_lane)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_topic, 10)

        # state
        self.last_imu_t = 0.0
        self.yaw_corr_deg: Optional[float] = None
        self.initial_heading_deg: Optional[float] = None

        self.last_lane_t = 0.0
        self.lane_conf: float = 0.0
        self.lane_has_pair: bool = False

        self.e_bottom: Optional[float] = None
        self.e_mid: Optional[float] = None
        self.ang_img: float = 0.0

        # desired offset
        self._desired_offset_deg: float = 0.0
        self._desired_offset_lpf: float = 0.0
        self._offset_rate_state: float = 0.0

        # yaw PID
        self._t_last = time.time()
        self._e_prev = 0.0
        self._i_state = 0.0

        # motor cmd
        self.cmd_l = 0
        self.cmd_r = 0

        dt = 1.0 / max(1.0, self.control_hz)
        self.timer = self.create_timer(dt, self.tick)

        self.get_logger().info("READY: lane_follow_node (lane1 nonlinear approach)")

    # ---------------- callbacks
    def cb_imu(self, msg: Imu):
        self.last_imu_t = time.time()
        q = msg.orientation
        yaw_raw = wrap_deg(rad2deg(yaw_from_quat(q.x, q.y, q.z, q.w)))
        self.yaw_corr_deg = wrap_deg(self.yaw_sign * yaw_raw)

        if self.initial_heading_deg is None:
            self.initial_heading_deg = float(self.yaw_corr_deg)
            self._desired_offset_deg = 0.0
            self._desired_offset_lpf = 0.0
            self._offset_rate_state = 0.0
            self._pid_reset(0.0)

    def _pick_lane1(self, data: dict) -> Optional[dict]:
        lane = data.get("lane1", None)
        valid = bool(data.get("valid_lane1", False))
        if lane is None:
            return None
        if self.require_valid_lane and not valid:
            return None
        return lane

    def cb_lane(self, msg: String):
        now = time.time()
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        self.last_lane_t = now

        lane1 = self._pick_lane1(data)
        if lane1 is None:
            self.lane_conf = 0.0
            self.lane_has_pair = False
            self.e_bottom = None
            self.e_mid = None
            self.ang_img = 0.0
            return

        conf = float(clamp(float(lane1.get("confidence", 0.0)), 0.0, 1.0))
        has_pair = bool(lane1.get("has_pair", False))

        e_b = float(lane1.get("bottom_intersect_x_norm", 0.0))
        e_m = float(lane1.get("mid_intersect_x_norm", 0.0)) if self.use_mid_term else 0.0
        ang = float(lane1.get("angle_to_red_signed_deg", 0.0))

        if abs(e_b) < self.lane_deadband:
            e_b = 0.0
        if abs(e_m) < self.lane_deadband:
            e_m = 0.0

        e_b = float(clamp(e_b, -1.0, 1.0))
        e_m = float(clamp(e_m, -1.0, 1.0))
        ang = float(clamp(ang, -90.0, 90.0))

        self.lane_conf = conf
        self.lane_has_pair = has_pair
        self.e_bottom = e_b
        self.e_mid = e_m
        self.ang_img = ang

        if self.initial_heading_deg is None:
            return

        # confidence gating
        g = 1.0
        if conf < self.conf_gate:
            g = clamp(conf / max(1e-6, self.conf_gate), 0.0, 1.0)
        if has_pair:
            g *= self.pair_boost

        # pre-shape raw signals
        eb_eff = e_b
        em_eff = e_m
        ang_eff = ang
        if self.soft_steer:
            eb_eff = softsign(e_b, self.offset_soft_k)
            em_eff = softsign(e_m, self.offset_soft_k)
            ang_eff = softsign(ang, self.angle_soft_k_deg) * self.angle_soft_k_deg

        # ---------------- NONLINEAR APPROACH LAW
        # Combine bottom+mid into a single "lateral error"
        e_mix = eb_eff
        if self.use_mid_term:
            e_mix = (1.0 - self.mid_mix) * eb_eff + self.mid_mix * em_eff

        if self.enable_nonlinear_approach:
            mag = abs(e_mix)  # 0..1
            approach = self.max_approach_deg * pow_ease(mag, self.approach_gamma)
            pos_term = sgn(e_mix) * approach
        else:
            # fallback: linear-ish
            pos_term = self.max_approach_deg * e_mix

        # small angle correction (keep it limited)
        ang_term = 0.0
        if self.use_lane_angle_term:
            ang_term = clamp(self.kp_ang * ang_eff, -self.max_angle_term_deg, self.max_angle_term_deg)

        desired = (pos_term + ang_term)
        desired = self.lane_sign * g * desired
        desired = float(clamp(desired, -self.yaw_offset_limit_deg, self.yaw_offset_limit_deg))
        self._desired_offset_deg = desired

    # ---------------- yaw PID
    def _pid_reset(self, e0: float):
        self._e_prev = e0
        self._i_state = 0.0

    def _yaw_pid_diff(self, dt: float, target_deg: float, yaw_deg: float):
        e = wrap_deg(target_deg - yaw_deg)
        de = (e - self._e_prev) / max(1e-3, dt)
        self._e_prev = e

        self._i_state += e * dt
        i_pwm = clamp(self.ki * self._i_state, -self.i_clamp_pwm, self.i_clamp_pwm)

        u = self.kp * e + i_pwm + self.kd * de
        u = clamp(u, -self.max_diff_pwm, self.max_diff_pwm)
        u *= self.turn_motor_sign
        return u, e

    # ---------------- motor helpers
    def publish_motor(self, l: int, r: int):
        m = Int16MultiArray()
        m.data = [int(clamp(l, -1000, 1000)), int(clamp(r, -1000, 1000))]
        self.pub_motor.publish(m)

    def motor_map(self, l: float, r: float):
        l2 = l * self.left_scale + self.left_trim
        r2 = r * self.right_scale + self.right_trim
        l2 = int(round(clamp(l2, -1000.0, 1000.0)))
        r2 = int(round(clamp(r2, -1000.0, 1000.0)))
        return l2, r2

    def slew_int(self, target: int, current: int) -> int:
        d = target - current
        if d > self.slew_per_tick:
            d = self.slew_per_tick
        elif d < -self.slew_per_tick:
            d = -self.slew_per_tick
        return current + d

    # ---------------- speed
    def _scaled(self, x: float) -> float:
        return float(x * clamp(self.speed_scale, 0.05, 2.0))

    def _compute_base(self) -> float:
        now = time.time()
        lane_age = now - self.last_lane_t

        if self.stop_if_lane_lost and lane_age > self.lane_timeout_s:
            return 0.0

        base = self._scaled(self.base_pwm)
        minb = self._scaled(self.min_base_pwm)
        maxb = self._scaled(self.max_base_pwm)

        # stale lane -> reduce
        if lane_age > self.lane_timeout_s:
            t = clamp((lane_age - self.lane_timeout_s) / max(1e-6, self.lane_timeout_s), 0.0, 1.0)
            base = lerp(base, minb, t)

        # low confidence -> reduce
        if self.lane_conf < 0.5:
            t = clamp((0.5 - self.lane_conf) / 0.5, 0.0, 1.0)
            base = lerp(base, minb, t)

        # optional slow on big errors (keep mild)
        if self.slow_on_error:
            e = abs(self.e_bottom) if self.e_bottom is not None else 0.0
            a = abs(self.ang_img)
            t1 = clamp(e / max(1e-6, self.slow_err_norm_at), 0.0, 1.0)
            t2 = clamp(a / max(1e-6, self.slow_angle_deg_at), 0.0, 1.0)
            t = max(t1, t2)
            ratio = lerp(1.0, self.slow_min_ratio, t)
            base = base * ratio

        return float(clamp(base, 0.0, maxb))

    def _apply_turn_boost(self, base: float, diff_abs: float) -> float:
        if not self.enable_turn_boost:
            return base

        start = max(0.0, self.turn_boost_start_pwm)
        full = max(start + 1e-6, self.turn_boost_full_pwm)

        t = (diff_abs - start) / (full - start)
        t = clamp(t, 0.0, 1.0)

        gain = lerp(1.0, max(1.0, self.turn_base_gain), t)
        boosted = base * gain

        cap = self._scaled(self.turn_boost_max_base_pwm)
        return float(clamp(boosted, 0.0, cap))

    # ---------------- main tick
    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self._t_last)
        self._t_last = now

        # IMU safety
        if (now - self.last_imu_t) > self.imu_timeout_s or self.yaw_corr_deg is None or self.initial_heading_deg is None:
            self.cmd_l = self.slew_int(0, self.cmd_l)
            self.cmd_r = self.slew_int(0, self.cmd_r)
            self.publish_motor(self.cmd_l, self.cmd_r)
            return

        yaw = float(self.yaw_corr_deg)

        # smooth desired offset (LPF + rate limit)
        self._desired_offset_lpf = lpf(self._desired_offset_lpf, self._desired_offset_deg, self.offset_lpf_alpha)

        max_step = self.desired_offset_rate_deg_s * dt
        d = self._desired_offset_lpf - self._offset_rate_state
        d = clamp(d, -max_step, max_step)
        self._offset_rate_state = float(clamp(self._offset_rate_state + d,
                                              -self.yaw_offset_limit_deg, self.yaw_offset_limit_deg))

        target = wrap_deg(self.initial_heading_deg + self._offset_rate_state)
        diff, e_yaw = self._yaw_pid_diff(dt, target, yaw)

        base = self._compute_base()
        base = self._apply_turn_boost(base, abs(diff))

        left = base + diff
        right = base - diff

        l_cmd, r_cmd = self.motor_map(left, right)
        self.cmd_l = self.slew_int(l_cmd, self.cmd_l)
        self.cmd_r = self.slew_int(r_cmd, self.cmd_r)
        self.publish_motor(self.cmd_l, self.cmd_r)

        # debug
        if self._dbg_period > 0.0 and (now - self._dbg_last) >= self._dbg_period:
            self._dbg_last = now
            lane_age = now - self.last_lane_t
            self.get_logger().info(
                f"yaw={yaw:+6.2f} tgt={target:+6.2f} eyaw={e_yaw:+5.2f} diff={diff:+6.1f} "
                f"lane_age={lane_age:.2f} eb={self.e_bottom} em={self.e_mid} ang={self.ang_img:+5.1f} "
                f"conf={self.lane_conf:.2f} base={base:.1f} off={self._offset_rate_state:+.2f}"
            )


def main():
    rclpy.init()
    node = LaneFollowNode()
    rclpy.spin(node)
    try:
        node.publish_motor(0, 0)
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()