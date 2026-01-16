#!/usr/bin/env python3
"""
Autonomous Step 0 (CPP lane_info compatible)

Davranış:
  - Yavaş ramp ile düz ileri başlar.
  - IMU ile başlangıç yönünü korur (heading hold).
  - Düz gidemiyorsa (yaw hata büyük ve uzun sürüyorsa) işlemi sonlandırır.
  - Lane görülünce, şeridin orta noktasını ekran orta yüksekliğine getirir ve durur.
  - Gerekirse küçük ileri/geri hamlelerle optimum merkezler.
    - Sonra komut bekler (left/right) ve şerit doğrultusuna döner.
"""

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


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ImuState:
    t: float
    yaw: float
    wz: float


class AutonomousStep0(Node):
    def __init__(self):
        super().__init__("autonomous_step_0")

        # Topics
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")
        self.declare_parameter("turn_cmd_topic", "/lane_turn/cmd")

        # Timing
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("imu_stale_s", 0.50)
        self.declare_parameter("lane_stale_s", 1.0)
        self.declare_parameter("shutdown_after_done_s", 0.3)

        # Straight drive
        self.declare_parameter("start_speed", 30.0)
        self.declare_parameter("cruise_speed", 80.0)
        self.declare_parameter("ramp_time_s", 2.0)
        self.declare_parameter("heading_kp", 280.0)
        self.declare_parameter("heading_kd", 12.0)
        self.declare_parameter("heading_ki", 0.0)
        self.declare_parameter("heading_steer_sign", -1.0)
        self.declare_parameter("wz_sign", -1.0)
        self.declare_parameter("max_heading_trim", 220.0)

        # Abort if cannot go straight
        self.declare_parameter("yaw_abort_deg", 12.0)
        self.declare_parameter("yaw_abort_hold_s", 1.0)

        # Lane detection / alignment
        self.declare_parameter("conf_enter", 0.55)
        self.declare_parameter("frames_confirm", 3)
        self.declare_parameter("frames_verify", 4)
        self.declare_parameter("dx_tol", 0.05)
        self.declare_parameter("y_tol_px", 8.0)
        self.declare_parameter("frames_settled", 6)
        self.declare_parameter("dx_sign", 1.0)
        self.declare_parameter("y_forward_sign", 1.0)
        self.declare_parameter("center_speed_max", 80.0)
        self.declare_parameter("center_k_dx", 180.0)
        self.declare_parameter("center_k_y", 0.8)
        self.declare_parameter("center_step_max", 60.0)

        # Turn control (post-align) - imu_pid_motion_node logic
        self.declare_parameter("angle_tol_deg", 3.0)
        self.declare_parameter("turn_pwm_min", 180.0)
        self.declare_parameter("turn_pwm_max", 380.0)
        self.declare_parameter("turn_kp", 6.0)
        self.declare_parameter("turn_ki", 0.0)
        self.declare_parameter("turn_kd", 18.0)
        self.declare_parameter("turn_settle_frames", 4)

        # Motor
        self.declare_parameter("slew_per_sec", 700.0)

        # Lane image size fallback
        self.declare_parameter("img_h", 240)

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
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)
        self.turn_cmd_topic = str(self.get_parameter("turn_cmd_topic").value)

        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos_fast)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.sub_cmd = self.create_subscription(String, self.turn_cmd_topic, self.cb_cmd, qos_fast)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        self.state = "INITIAL"  # INITIAL -> RAMP -> WAIT_LANE -> VERIFY_LANE -> ALIGN -> TURN_WAIT -> TURNING -> DONE -> ABORT
        self.turn_dir: Optional[str] = None
        self.imu: Optional[ImuState] = None
        self._yaw_ref = 0.0
        self._yaw_err_i = 0.0
        self._bias = 0.0
        self._last_step_t = time.time()
        self._last_turn_err = None
        self._last_turn_t = None
        self._turn_settle = 0
        self._turn_err_i = 0.0

        self.lane_data = None
        self.lane_last_t = 0.0
        self.conf_count = 0
        self.verify_count = 0
        self.settle_count = 0

        self._abort_start: Optional[float] = None

        self._last_cmd_l = 0.0
        self._last_cmd_r = 0.0
        self._last_cmd_time = time.time()

        self._done_triggered = False

        hz = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1.0, hz), self.step)

        self.get_logger().info(
            f"[READY] step0: lane_topic={self.lane_info_topic} imu_topic={self.imu_topic}"
        )

    # ========== IMU ==========
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
        self.imu = ImuState(t=now, yaw=yaw, wz=wz)

    # ========== Lane ==========
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
            if self.state == "TURN_WAIT":
                self.state = "TURNING"
            self.get_logger().info(f"[CMD] turn_dir={self.turn_dir}")
        elif s in ("stop", "cancel"):
            self.turn_dir = None
            if self.state in ("TURNING", "TURN_WAIT"):
                self.state = "TURN_WAIT"
            self.publish_motor(0.0, 0.0)

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

    def _get_selected_center(self) -> Optional[dict]:
        if not isinstance(self.lane_data, dict):
            return None
        slc = self.lane_data.get("selected_line_center")
        if isinstance(slc, dict):
            return slc
        return None

    # ========== Control ==========
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
        trim = clamp(trim * steer_sign, -float(self.get_parameter("max_heading_trim").value),
                     float(self.get_parameter("max_heading_trim").value))
        return trim

    def _should_abort(self, dt: float) -> bool:
        if self.imu is None:
            return False
        yaw_abort_deg = float(self.get_parameter("yaw_abort_deg").value)
        hold_s = float(self.get_parameter("yaw_abort_hold_s").value)
        yaw_err = abs(rad2deg(wrap_pi(self._yaw_ref - self.imu.yaw)))
        if yaw_err >= yaw_abort_deg:
            if self._abort_start is None:
                self._abort_start = time.time()
            elif (time.time() - self._abort_start) >= hold_s:
                return True
        else:
            self._abort_start = None
        return False

    def step(self):
        now = time.time()
        dt = max(0.001, now - self._last_step_t)
        self._last_step_t = now

        # IMU freshness
        if self.imu is None or (now - self.imu.t) > float(self.get_parameter("imu_stale_s").value):
            self.publish_motor(0.0, 0.0)
            return

        if self.state == "INITIAL":
            self._yaw_ref = self.imu.yaw
            self.state = "RAMP"
            self._ramp_start = now
            self.get_logger().info(f"[INITIAL->RAMP] yaw_ref={rad2deg(self._yaw_ref):.1f}°")

        elif self.state == "RAMP":
            if self._should_abort(dt):
                self.state = "ABORT"
            else:
                cmd_l, cmd_r = self._drive_straight(ramp=True, dt=dt)
                self.publish_motor(cmd_l, cmd_r)
                if (now - self._ramp_start) >= float(self.get_parameter("ramp_time_s").value):
                    self.state = "WAIT_LANE"

        elif self.state == "WAIT_LANE":
            if self._should_abort(dt):
                self.state = "ABORT"
            else:
                cmd_l, cmd_r = self._drive_straight(ramp=False, dt=dt)
                self.publish_motor(cmd_l, cmd_r)
                if self._lane_seen():
                    self.state = "VERIFY_LANE"
                    self.verify_count = 0
                    self.settle_count = 0
                    self.get_logger().info("[WAIT_LANE->VERIFY_LANE] lane detected, verifying")

        elif self.state == "VERIFY_LANE":
            # stop and verify a few frames
            self.publish_motor(0.0, 0.0)
            if not self._lane_valid():
                self.state = "WAIT_LANE"
                self.get_logger().info("[VERIFY_LANE->WAIT_LANE] lane lost")
            elif self._lane_conf_good():
                self.verify_count += 1
                if self.verify_count >= int(self.get_parameter("frames_verify").value):
                    self.state = "ALIGN"
                    self.settle_count = 0
                    self.get_logger().info("[VERIFY_LANE->ALIGN] lane confirmed")
            else:
                self.state = "WAIT_LANE"
                self.get_logger().info("[VERIFY_LANE->WAIT_LANE] confidence dropped")

        elif self.state == "ALIGN":
            if not self._lane_valid():
                self.state = "WAIT_LANE"
            else:
                cmd_l, cmd_r, settled = self._align_step(dt)
                self.publish_motor(cmd_l, cmd_r)
                if settled:
                    self.state = "TURN_WAIT"
                    self.get_logger().info("[ALIGN->TURN_WAIT] lane centered, waiting for turn cmd")

        elif self.state == "TURN_WAIT":
            self.publish_motor(0.0, 0.0)
            # wait for command in cb_cmd

        elif self.state == "TURNING":
            if self.turn_dir is None:
                self.state = "TURN_WAIT"
            else:
                cmd_l, cmd_r, done = self._turn_step()
                self.publish_motor(cmd_l, cmd_r)
                if done:
                    self.state = "DONE"

        elif self.state == "ABORT":
            self.publish_motor(0.0, 0.0)
            self.get_logger().error("[ABORT] Unable to hold straight heading")
            self._shutdown_once()

        elif self.state == "DONE":
            self.publish_motor(0.0, 0.0)
            self.get_logger().info("Step 0 COMPLETE")

    def _lane_seen(self) -> bool:
        if not self._lane_valid():
            self.conf_count = 0
            return False
        sel = self._get_selected()
        if sel is None:
            self.conf_count = 0
            return False
        conf_enter = float(self.get_parameter("conf_enter").value)
        if float(sel.get("confidence", 0.0)) >= conf_enter:
            self.conf_count += 1
            if self.conf_count >= int(self.get_parameter("frames_confirm").value):
                self.conf_count = 0
                return True
        else:
            self.conf_count = 0
        return False

    def _lane_conf_good(self) -> bool:
        if not self._lane_valid():
            return False
        sel = self._get_selected()
        if sel is None:
            return False
        conf_enter = float(self.get_parameter("conf_enter").value)
        return float(sel.get("confidence", 0.0)) >= conf_enter

    def _get_lane_angle_deg(self) -> Optional[float]:
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

    def _drive_straight(self, ramp: bool, dt: float):
        start_speed = float(self.get_parameter("start_speed").value)
        cruise_speed = float(self.get_parameter("cruise_speed").value)
        if ramp:
            t = (time.time() - self._ramp_start) / max(0.1, float(self.get_parameter("ramp_time_s").value))
            speed = start_speed + (cruise_speed - start_speed) * clamp(t, 0.0, 1.0)
        else:
            speed = cruise_speed

        trim = self._heading_trim(dt)

        # auto bias compensation for asymmetric drivetrain (imu_motion_test style)
        if self.imu is not None:
            yaw_err = wrap_pi(self._yaw_ref - self.imu.yaw)
            self._bias += clamp(yaw_err, -0.06, 0.06) * 60.0 * dt
            self._bias = clamp(self._bias, -120.0, 120.0)

        cmd_l = speed + trim - self._bias
        cmd_r = speed - trim + self._bias
        return cmd_l, cmd_r

    def _align_step(self, dt: float):
        slc = self._get_selected_center()
        sel = self._get_selected()
        if slc is None or sel is None:
            return 0.0, 0.0, False

        dx_norm = float(slc.get("dx_norm", 0.0)) * float(self.get_parameter("dx_sign").value)
        img_h = int(self.lane_data.get("img", {}).get("h", self.get_parameter("img_h").value))
        y_px = float(slc.get("y_px", img_h / 2.0))
        y_err = y_px - (img_h * 0.5)

        dx_tol = float(self.get_parameter("dx_tol").value)
        y_tol = float(self.get_parameter("y_tol_px").value)
        k_dx = float(self.get_parameter("center_k_dx").value)
        k_y = float(self.get_parameter("center_k_y").value)
        center_step_max = float(self.get_parameter("center_step_max").value)
        center_speed_max = float(self.get_parameter("center_speed_max").value)

        # forward/back based on vertical error
        forward_sign = float(self.get_parameter("y_forward_sign").value)
        fwd = clamp(-forward_sign * k_y * y_err, -center_step_max, center_step_max)

        # lateral trim using dx
        steer = clamp(k_dx * dx_norm, -center_speed_max, center_speed_max)

        # heading hold (small) to keep straight
        heading_trim = clamp(self._heading_trim(dt), -120.0, 120.0)

        cmd_l = fwd + steer + heading_trim
        cmd_r = fwd - steer - heading_trim

        if abs(dx_norm) <= dx_tol and abs(y_err) <= y_tol:
            self.settle_count += 1
            if self.settle_count >= int(self.get_parameter("frames_settled").value):
                return 0.0, 0.0, True
        else:
            self.settle_count = 0

        return cmd_l, cmd_r, False

    def _turn_step(self):
        ang = self._get_lane_angle_deg()
        if ang is None:
            return 0.0, 0.0, False

        ang_abs = abs(ang)
        now = time.time()
        tol = float(self.get_parameter("angle_tol_deg").value)
        settle_frames = int(self.get_parameter("turn_settle_frames").value)

        if ang_abs <= tol:
            self._turn_settle += 1
            if self._turn_settle >= settle_frames:
                return 0.0, 0.0, True
        else:
            self._turn_settle = 0

        # PID on signed error (deg)
        sign = -1.0 if self.turn_dir == "left" else 1.0
        err = sign * ang_abs

        if self._last_turn_err is None or self._last_turn_t is None:
            derr = 0.0
        else:
            dt = max(0.001, now - self._last_turn_t)
            derr = (err - self._last_turn_err) / dt
        self._last_turn_err = err
        self._last_turn_t = now

        kp = float(self.get_parameter("turn_kp").value)
        ki = float(self.get_parameter("turn_ki").value)
        kd = float(self.get_parameter("turn_kd").value)

        if ki != 0.0:
            self._turn_err_i += err * 0.02
            self._turn_err_i = clamp(self._turn_err_i, -0.6, 0.6)
        else:
            self._turn_err_i = 0.0

        pwm = kp * abs(err) + ki * abs(self._turn_err_i) - kd * abs(derr)
        pwm = clamp(
            pwm,
            float(self.get_parameter("turn_pwm_min").value),
            float(self.get_parameter("turn_pwm_max").value),
        )

        # apply bias learned during step0 to keep rotation consistent
        if err > 0:
            cmd_l, cmd_r = pwm + self._bias, -pwm - self._bias
        else:
            cmd_l, cmd_r = -pwm - self._bias, pwm + self._bias

        return cmd_l, cmd_r, False

    def _shutdown_once(self):
        if self._done_triggered:
            return
        self._done_triggered = True
        try:
            self.timer.cancel()
        except Exception:
            pass
        delay = float(self.get_parameter("shutdown_after_done_s").value)
        self.create_timer(delay, lambda: rclpy.shutdown() if rclpy.ok() else None)

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
    node = AutonomousStep0()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()