#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Imu


# =========================
# utils
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


# =========================
# obs
# =========================
@dataclass
class LaneObs:
    t: float
    dx_norm: Optional[float]   # [-1..+1] left:- right:+
    ang_deg: Optional[float]   # signed deg (lane angle)
    conf: float


@dataclass
class ImuObs:
    t: float
    yaw: float
    wz: float  # rad/s (signed, after wz_sign)


# =========================
# node
# =========================
class LaneFollowImu(Node):
    """
    Basit ve deterministik 3-adım mantık:

    1) ALIGN (Açı düzelt):
       |ang| >= ang_align_enter_deg ise:
         IMU ile hedef yöne dön: target = yaw_now + (-ang_sign * ang_rad)
         Dönüş bitince (|ang|<=ang_align_exit_deg ve yaw_err küçük) => CENTER

    2) CENTER (Ortala):
       |dx| >= dx_center_enter ise:
         ileri giderek (gerekirse çok büyük dx'de yavaş) dx'yi 0'a yaklaştır.
         steer = k_center * dx (sign ile) sadece küçük trim.
         |dx| <= dx_center_exit olunca => TRACK

    3) TRACK (Düz git + küçük düzelt):
       küçük dx ve küçük ang varken ileri git, küçük yaw trim uygula.

    Not:
    - Center modda geri gitme yok. Sadece ileri + küçük trim. (Senin "rezalet" dediğin kısım buydu.)
    - Büyük açı varsa kesinlikle sadece dön (ALIGN). Büyük açı bitmeden center/track'e karışmıyor.
    """

    def __init__(self):
        super().__init__("lane_follow_imu_simple")

        # -------- topics
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("user_cmd_topic", "/lane_follow/cmd")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        # -------- staleness
        self.declare_parameter("imu_stale_s", 0.60)
        self.declare_parameter("lane_stale_s", 1.00)
        self.declare_parameter("behavior_on_lane_stale", "slow")  # stop|slow|keep
        self.declare_parameter("stale_slow_speed", 60.0)

        # -------- IMU sign
        # Senin ölçüm: sağa çevirince angular_velocity.z NEGATIF => wz_sign=+1 doğru
        self.declare_parameter("wz_sign", 1.0)

        # motor mixing sign (gerekirse)
        self.declare_parameter("steer_sign", 1.0)

        # -------- lane sign
        self.declare_parameter("dx_sign", -1.0)   # dx sağ(+): sola dön
        self.declare_parameter("ang_sign", 1.0)   # angle işareti tersse -1

        # -------- confidence / filtering
        self.declare_parameter("min_conf", 0.10)
        self.declare_parameter("dx_ema_alpha", 0.25)
        self.declare_parameter("ang_ema_alpha", 0.25)
        self.declare_parameter("max_dx_jump", 0.80)
        self.declare_parameter("max_ang_jump_deg", 55.0)

        # -------- yaw control
        self.declare_parameter("k_yaw_p", 190.0)
        self.declare_parameter("k_yaw_d", 55.0)
        self.declare_parameter("max_steer", 260.0)

        # -------- yaw target rate limit (3 FPS stabilite)
        self.declare_parameter("yaw_target_rate_rad_s", 0.9)

        # -------- speeds
        self.declare_parameter("speed_track", 140.0)        # düz giderken
        self.declare_parameter("speed_center", 120.0)       # ortalarken
        self.declare_parameter("speed_align_min", 150.0)    # dönerken min
        self.declare_parameter("speed_align_max", 240.0)    # dönerken max
        self.declare_parameter("speed_align_k", 170.0)      # |yaw_err| rad başına ek hız

        self.declare_parameter("speed_min", 0.0)
        self.declare_parameter("speed_max", 320.0)

        # -------- ALIGN thresholds
        self.declare_parameter("ang_align_enter_deg", 15.0)
        self.declare_parameter("ang_align_exit_deg", 7.0)
        self.declare_parameter("align_finish_yaw_err_deg", 4.0)
        self.declare_parameter("align_min_time_s", 0.20)
        self.declare_parameter("align_timeout_s", 2.8)
        self.declare_parameter("align_plan_gain", 1.0)      # 15deg ise 15deg dön

        # -------- CENTER thresholds
        self.declare_parameter("dx_center_enter", 0.18)
        self.declare_parameter("dx_center_exit", 0.08)
        self.declare_parameter("center_trim_rad_max", 0.35) # center sırasında yaw trim limiti
        self.declare_parameter("k_center_trim", 0.55)       # trim_rad = k * dx (sign dahil)

        # -------- TRACK trims
        self.declare_parameter("k_track_dx_trim", 0.35)     # rad per dx
        self.declare_parameter("k_track_ang_trim", 0.70)    # rad per ang_rad
        self.declare_parameter("track_trim_rad_max", 0.45)

        # -------- motor shaping
        self.declare_parameter("slew_per_sec", 520.0)
        self.declare_parameter("right_bias", 0.0)

        # -------- control
        self.declare_parameter("control_hz", 30.0)

        # ================= state
        self.running = False
        self.mode = "TRACK"  # TRACK | CENTER | ALIGN

        self.latest_lane: Optional[LaneObs] = None
        self.latest_imu: Optional[ImuObs] = None

        self._yaw = 0.0
        self._last_imu_t: Optional[float] = None

        self.dx_f = 0.0
        self.ang_f_rad = 0.0
        self.have_dx = False
        self.have_ang = False
        self._last_lane_good_t = 0.0

        # yaw target smoothing
        self._yaw_target = 0.0
        self._last_step_t = time.time()

        # align state
        self.align_start_t = 0.0
        self.align_yaw_target = 0.0

        # motor slew
        self._cmd_l = 0.0
        self._cmd_r = 0.0
        self._last_cmd_time = time.time()

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

        self.lane_info_topic = str(self.get_parameter("lane_info_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.user_cmd_topic = str(self.get_parameter("user_cmd_topic").value)
        self.motor_cmd_topic = str(self.get_parameter("motor_cmd_topic").value)

        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos_fast)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_fast)
        self.sub_cmd = self.create_subscription(String, self.user_cmd_topic, self.cb_cmd, qos_fast)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, qos_motor)

        hz = float(self.get_parameter("control_hz").value)
        self.timer = self.create_timer(1.0 / max(5.0, hz), self.step)

        self._last_dbg_t = 0.0
        self.get_logger().info("lane_follow_imu_simple READY")

    # ---------------- cmd
    def cb_cmd(self, msg: String):
        s = (msg.data or "").strip().lower()
        if s == "forward":
            self.running = True
            self.mode = "TRACK"
            if self.latest_imu is not None:
                self._yaw_target = self.latest_imu.yaw
            self.get_logger().info("CMD forward -> running")
        elif s == "stop":
            self.running = False
            self.mode = "TRACK"
            self.publish_motor(0.0, 0.0, force=True)
            self.get_logger().info("CMD stop -> stopped")
        elif s in ("reset", "reset_yaw"):
            if self.latest_imu is not None:
                self._yaw_target = self.latest_imu.yaw
            self.mode = "TRACK"
            self.get_logger().info("CMD reset_yaw -> yaw_target reset")

    # ---------------- lane parse
    def _parse_dx(self, sel: dict) -> Optional[float]:
        if not isinstance(sel, dict):
            return None

        # prefer center_offset_x_norm
        v = sel.get("center_offset_x_norm", None)
        if v is not None:
            try:
                return clamp(float(v), -1.0, 1.0)
            except Exception:
                pass

        # fallback bottom_intersect_x_norm
        v = sel.get("bottom_intersect_x_norm", None)
        if v is not None:
            try:
                x = float(v)
                if 0.0 <= x <= 1.0:
                    return clamp((x - 0.5) * 2.0, -1.0, 1.0)
                return clamp(x, -1.0, 1.0)
            except Exception:
                pass

        return None

    def cb_lane(self, msg: String):
        now = time.time()
        try:
            d = json.loads(msg.data)
        except Exception:
            return

        sel = d.get("selected", None) or d.get("lane1", None)

        dx = None
        ang_deg = None
        conf = 0.0

        if isinstance(sel, dict):
            dx = self._parse_dx(sel)

            a = sel.get("angle_to_red_signed_deg", None)
            if a is not None:
                try:
                    ang_deg = float(a)
                except Exception:
                    ang_deg = None

            c = sel.get("confidence", None)
            if c is not None:
                try:
                    conf = float(c)
                except Exception:
                    conf = 0.0

        self.latest_lane = LaneObs(t=now, dx_norm=dx, ang_deg=ang_deg, conf=conf)

        if conf < float(self.get_parameter("min_conf").value):
            return

        # dx filter
        if dx is not None:
            if not self.have_dx:
                self.dx_f = dx
                self.have_dx = True
            else:
                if abs(dx - self.dx_f) <= float(self.get_parameter("max_dx_jump").value):
                    a = float(self.get_parameter("dx_ema_alpha").value)
                    self.dx_f = (1.0 - a) * self.dx_f + a * dx

        # angle filter
        if ang_deg is not None:
            ang_deg = clamp(ang_deg, -90.0, 90.0)
            ang_rad = deg2rad(ang_deg)
            if not self.have_ang:
                self.ang_f_rad = ang_rad
                self.have_ang = True
            else:
                prev_deg = rad2deg(self.ang_f_rad)
                if abs(ang_deg - prev_deg) <= float(self.get_parameter("max_ang_jump_deg").value):
                    a = float(self.get_parameter("ang_ema_alpha").value)
                    self.ang_f_rad = (1.0 - a) * self.ang_f_rad + a * ang_rad

        self._last_lane_good_t = now

    # ---------------- imu
    def cb_imu(self, msg: Imu):
        now = time.time()
        wz_sign = float(self.get_parameter("wz_sign").value)
        wz = wz_sign * float(msg.angular_velocity.z)

        if self._last_imu_t is None:
            self._last_imu_t = now
        dt = clamp(now - self._last_imu_t, 0.0, 0.2)
        self._last_imu_t = now

        self._yaw = wrap_pi(self._yaw + wz * dt)
        self.latest_imu = ImuObs(t=now, yaw=self._yaw, wz=wz)

    # =========================
    # ALIGN plan
    # =========================
    def _start_align(self, now: float, imu: ImuObs):
        ang_sign = float(self.get_parameter("ang_sign").value)
        gain = float(self.get_parameter("align_plan_gain").value)

        # "15 derece ise 15 derece dön"
        desired_turn = (-ang_sign) * self.ang_f_rad * gain
        desired_turn = clamp(desired_turn, -math.radians(85.0), math.radians(85.0))

        self.align_start_t = now
        self.align_yaw_target = wrap_pi(imu.yaw + desired_turn)
        self.mode = "ALIGN"

    # =========================
    # main loop
    # =========================
    def step(self):
        now = time.time()

        # IMU fresh?
        imu_ok = self.latest_imu is not None and (now - self.latest_imu.t) <= float(self.get_parameter("imu_stale_s").value)
        if not imu_ok:
            self.running = False
            self.mode = "TRACK"
            self.publish_motor(0.0, 0.0, force=True)
            return

        if not self.running:
            self.publish_motor(0.0, 0.0)
            return

        imu = self.latest_imu
        assert imu is not None

        # lane fresh?
        lane_ok = (now - self._last_lane_good_t) <= float(self.get_parameter("lane_stale_s").value)
        if not lane_ok:
            behavior = str(self.get_parameter("behavior_on_lane_stale").value).strip().lower()
            if behavior == "stop":
                self.publish_motor(0.0, 0.0)
                return
            elif behavior == "slow":
                base = float(self.get_parameter("stale_slow_speed").value)
            else:
                base = float(self.get_parameter("speed_track").value)

            base = clamp(base, float(self.get_parameter("speed_min").value), float(self.get_parameter("speed_max").value))
            # stale iken düz gitmeye yakın tut
            yaw_target_des = imu.yaw
            self._apply_and_publish(now, imu, yaw_target_des, base)
            return

        # lane values
        ang_deg = rad2deg(self.ang_f_rad) if self.have_ang else 0.0
        dx = self.dx_f if self.have_dx else 0.0

        # ---------------- mode decide
        ang_enter = float(self.get_parameter("ang_align_enter_deg").value)
        ang_exit = float(self.get_parameter("ang_align_exit_deg").value)

        dx_enter = float(self.get_parameter("dx_center_enter").value)
        dx_exit = float(self.get_parameter("dx_center_exit").value)

        # If big angle -> ALIGN has priority
        if self.have_ang and abs(ang_deg) >= ang_enter:
            if self.mode != "ALIGN":
                self._start_align(now, imu)

        # If currently ALIGN, check exit/timeout
        if self.mode == "ALIGN":
            yaw_err_deg = abs(rad2deg(wrap_pi(self.align_yaw_target - imu.yaw)))
            finish_yaw_err = float(self.get_parameter("align_finish_yaw_err_deg").value)
            min_time = float(self.get_parameter("align_min_time_s").value)
            timeout = float(self.get_parameter("align_timeout_s").value)

            if (now - self.align_start_t) >= min_time:
                # Exit if lane angle small enough AND yaw_err small enough
                if (self.have_ang and abs(ang_deg) <= ang_exit and yaw_err_deg <= finish_yaw_err):
                    self.mode = "CENTER" if (self.have_dx and abs(dx) >= dx_enter) else "TRACK"
                elif (now - self.align_start_t) >= timeout:
                    self.mode = "CENTER" if (self.have_dx and abs(dx) >= dx_enter) else "TRACK"

        # If not ALIGN, decide CENTER vs TRACK
        if self.mode != "ALIGN":
            if self.have_dx and abs(dx) >= dx_enter:
                self.mode = "CENTER"
            elif self.mode == "CENTER" and self.have_dx and abs(dx) <= dx_exit:
                self.mode = "TRACK"
            elif self.mode not in ("CENTER", "TRACK"):
                self.mode = "TRACK"

        # ---------------- act by mode
        dx_sign = float(self.get_parameter("dx_sign").value)
        ang_sign = float(self.get_parameter("ang_sign").value)

        if self.mode == "ALIGN":
            # yaw target is fixed planned yaw
            yaw_target_des = self.align_yaw_target

            yaw_err_rad = abs(wrap_pi(yaw_target_des - imu.yaw))
            base = float(self.get_parameter("speed_align_min").value) + float(self.get_parameter("speed_align_k").value) * yaw_err_rad
            base = clamp(base,
                         float(self.get_parameter("speed_align_min").value),
                         float(self.get_parameter("speed_align_max").value))

        elif self.mode == "CENTER":
            # Move forward and trim yaw based ONLY on dx
            base = float(self.get_parameter("speed_center").value)

            trim = (dx_sign * dx) * float(self.get_parameter("k_center_trim").value)
            trim = clamp(trim, -float(self.get_parameter("center_trim_rad_max").value),
                         float(self.get_parameter("center_trim_rad_max").value))
            yaw_target_des = wrap_pi(imu.yaw + trim)

            base = clamp(base, float(self.get_parameter("speed_min").value), float(self.get_parameter("speed_max").value))

        else:
            # TRACK: forward, small trim by dx + angle
            base = float(self.get_parameter("speed_track").value)

            trim = 0.0
            if self.have_dx:
                trim += dx_sign * dx * float(self.get_parameter("k_track_dx_trim").value)
            if self.have_ang:
                trim += (-ang_sign) * self.ang_f_rad * float(self.get_parameter("k_track_ang_trim").value)

            trim = clamp(trim, -float(self.get_parameter("track_trim_rad_max").value),
                         float(self.get_parameter("track_trim_rad_max").value))

            yaw_target_des = wrap_pi(imu.yaw + trim)
            base = clamp(base, float(self.get_parameter("speed_min").value), float(self.get_parameter("speed_max").value))

        self._apply_and_publish(now, imu, yaw_target_des, base)

        # debug 1Hz
        if now - self._last_dbg_t >= 1.0:
            self._last_dbg_t = now
            self.get_logger().info(
                f"mode={self.mode} dx={dx:.3f} ang_deg={ang_deg:.1f} base={base:.1f} yaw={imu.yaw:.2f} tgt={self._yaw_target:.2f} wz={imu.wz:.3f}"
            )

    # =========================
    # apply: yaw target rate limit + PD + publish
    # =========================
    def _apply_and_publish(self, now: float, imu: ImuObs, yaw_target_des: float, base: float):
        # yaw target rate limit
        dt = max(1e-3, now - self._last_step_t)
        self._last_step_t = now
        rate = float(self.get_parameter("yaw_target_rate_rad_s").value)
        max_step = rate * dt
        err_t = wrap_pi(yaw_target_des - self._yaw_target)
        self._yaw_target = wrap_pi(self._yaw_target + clamp(err_t, -max_step, max_step))

        # yaw PD
        kp = float(self.get_parameter("k_yaw_p").value)
        kd = float(self.get_parameter("k_yaw_d").value)
        max_steer = float(self.get_parameter("max_steer").value)

        yaw_err = wrap_pi(self._yaw_target - imu.yaw)
        steer = kp * yaw_err - kd * imu.wz
        steer = clamp(steer, -max_steer, max_steer)
        steer *= float(self.get_parameter("steer_sign").value)

        # mix
        l = base - steer
        r = base + steer
        rb = float(self.get_parameter("right_bias").value)
        r = r + rb if r >= 0.0 else r - rb

        self.publish_motor(l, r)

    # =========================
    # motor publish (slew)
    # =========================
    def publish_motor(self, l: float, r: float, force: bool = False):
        now = time.time()
        dt = max(1e-3, now - self._last_cmd_time)
        self._last_cmd_time = now

        slew = float(self.get_parameter("slew_per_sec").value)
        max_step = slew * dt

        l = clamp(l, -1000.0, 1000.0)
        r = clamp(r, -1000.0, 1000.0)

        if force:
            self._cmd_l = l
            self._cmd_r = r
        else:
            self._cmd_l += clamp(l - self._cmd_l, -max_step, max_step)
            self._cmd_r += clamp(r - self._cmd_r, -max_step, max_step)

        msg = Int16MultiArray()
        msg.data = [int(round(self._cmd_l)), int(round(self._cmd_r))]
        self.pub_motor.publish(msg)


def main():
    rclpy.init()
    node = LaneFollowImu()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()