#!/usr/bin/env python3
import math
import time
from enum import Enum
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from std_msgs.msg import Int16MultiArray


def wrap_pi(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw (Z axis) from quaternion (assumes ENU-ish frame: Z up)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clampi(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


@dataclass
class ImuSample:
    stamp_s: float = 0.0
    yaw: float = 0.0
    gz: float = 0.0  # rad/s (angular_velocity.z)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0


class State(Enum):
    WAIT_IMU = 0
    MOTOR_TEST = 1
    CALIB = 2
    FWD1 = 3
    TURN_RIGHT = 4
    FWD2 = 5
    DONE = 6


@dataclass
class MotorStep:
    name: str
    left: int
    right: int
    dur_s: float


class ImuMotionTestNode(Node):
    def __init__(self):
        super().__init__("imu_motion_test_node")

        # ---------------- Params ----------------
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_topic", "/motor_cmd")
        self.declare_parameter("rate_hz", 50.0)

        # Motor test enable
        self.declare_parameter("motor_test_enable", True)
        self.declare_parameter("motor_test_pwm", 450)
        self.declare_parameter("motor_test_step_s", 1.2)
        self.declare_parameter("motor_test_pause_s", 0.4)

        # IMU sanity log
        self.declare_parameter("imu_log_hz", 5.0)   # how often log IMU (Hz)
        self.declare_parameter("use_header_stamp", False)  # True: stamp from msg.header if valid

        # Sequence (after motor test/calib)
        self.declare_parameter("t_forward_1", 1.0)
        self.declare_parameter("t_forward_2", 0.5)
        self.declare_parameter("turn_deg", 90.0)
        self.declare_parameter("turn_timeout_s", 4.0)

        # PWM targets
        self.declare_parameter("fwd_target_pwm", 650)      # 0..1000
        self.declare_parameter("turn_target_pwm", 650)     # 0..1000

        # Ramping / stiction
        self.declare_parameter("pwm_min_start", 350)
        self.declare_parameter("pwm_step", 25)
        self.declare_parameter("pwm_max", 1000)

        # Movement detection thresholds
        self.declare_parameter("gyro_move_thr_dps", 8.0)   # deg/s
        self.declare_parameter("yaw_move_thr_deg", 1.0)    # deg

        # Straight heading hold (PD)
        self.declare_parameter("straight_kp", 220.0)
        self.declare_parameter("straight_kd", 30.0)
        self.declare_parameter("straight_max_delta", 220)
        self.declare_parameter("straight_trim", 0.0)

        # Turn control (PD)
        self.declare_parameter("turn_kp", 520.0)
        self.declare_parameter("turn_kd", 40.0)
        self.declare_parameter("yaw_tol_deg", 3.0)

        # Calibration enable
        self.declare_parameter("calib_enable", True)
        self.declare_parameter("calib_spin_s", 0.8)
        self.declare_parameter("calib_pause_s", 0.25)
        self.declare_parameter("calib_fwd_s", 1.2)

        # Turn sign hint
        self.declare_parameter("turn_sign_hint", 0)        # 0=auto, +1/-1=force

        # ---------------- Read params ----------------
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))

        self.motor_test_enable = bool(self.get_parameter("motor_test_enable").value)
        self.motor_test_pwm = clampi(int(self.get_parameter("motor_test_pwm").value), 0, 1000)
        self.motor_test_step_s = float(self.get_parameter("motor_test_step_s").value)
        self.motor_test_pause_s = float(self.get_parameter("motor_test_pause_s").value)

        self.imu_log_hz = max(0.5, float(self.get_parameter("imu_log_hz").value))
        self.use_header_stamp = bool(self.get_parameter("use_header_stamp").value)

        self.t_fwd1 = float(self.get_parameter("t_forward_1").value)
        self.t_fwd2 = float(self.get_parameter("t_forward_2").value)
        self.turn_rad = math.radians(float(self.get_parameter("turn_deg").value))
        self.turn_timeout_s = float(self.get_parameter("turn_timeout_s").value)
        self.yaw_tol = math.radians(float(self.get_parameter("yaw_tol_deg").value))

        self.fwd_target_pwm = clampi(int(self.get_parameter("fwd_target_pwm").value), 0, 1000)
        self.turn_target_pwm = clampi(int(self.get_parameter("turn_target_pwm").value), 0, 1000)

        self.pwm_min_start = clampi(int(self.get_parameter("pwm_min_start").value), 0, 1000)
        self.pwm_step = max(1, int(self.get_parameter("pwm_step").value))
        self.pwm_max = clampi(int(self.get_parameter("pwm_max").value), 0, 1000)

        self.gyro_thr = math.radians(float(self.get_parameter("gyro_move_thr_dps").value))
        self.yaw_move_thr = math.radians(float(self.get_parameter("yaw_move_thr_deg").value))

        self.straight_kp = float(self.get_parameter("straight_kp").value)
        self.straight_kd = float(self.get_parameter("straight_kd").value)
        self.straight_max_delta = int(self.get_parameter("straight_max_delta").value)
        self.straight_trim = float(self.get_parameter("straight_trim").value)

        self.turn_kp = float(self.get_parameter("turn_kp").value)
        self.turn_kd = float(self.get_parameter("turn_kd").value)

        self.calib_enable = bool(self.get_parameter("calib_enable").value)
        self.calib_spin_s = float(self.get_parameter("calib_spin_s").value)
        self.calib_pause_s = float(self.get_parameter("calib_pause_s").value)
        self.calib_fwd_s = float(self.get_parameter("calib_fwd_s").value)

        self.turn_sign_hint = int(self.get_parameter("turn_sign_hint").value)

        # ---------------- State ----------------
        self.state = State.WAIT_IMU
        self.have_imu = False
        self.imu = ImuSample()

        self.t_state0 = self.now_s()
        self.heading_ref = 0.0
        self.yaw_start = 0.0
        self.yaw_target = 0.0

        self.turn_sign = 0
        self.min_turn_pwm = None

        # Motor test steps
        p = self.motor_test_pwm
        step = self.motor_test_step_s
        pause = self.motor_test_pause_s
        self.motor_steps = [
            MotorStep("PAUSE", 0, 0, pause),

            MotorStep("LEFT_FWD", +p, 0, step),
            MotorStep("PAUSE", 0, 0, pause),
            MotorStep("RIGHT_FWD", 0, +p, step),
            MotorStep("PAUSE", 0, 0, pause),

            MotorStep("LEFT_REV", -p, 0, step),
            MotorStep("PAUSE", 0, 0, pause),
            MotorStep("RIGHT_REV", 0, -p, step),
            MotorStep("PAUSE", 0, 0, pause),

            MotorStep("BOTH_FWD", +p, +p, step),
            MotorStep("PAUSE", 0, 0, pause),
            MotorStep("BOTH_REV", -p, -p, step),
            MotorStep("PAUSE", 0, 0, pause),

            MotorStep("TANK_TURN(+,-)", +p, -p, step),
            MotorStep("PAUSE", 0, 0, pause),
            MotorStep("TANK_TURN(-,+)", -p, +p, step),
            MotorStep("PAUSE", 0, 0, pause),
        ]
        self.motor_step_i = 0
        self.motor_step_started = False

        # ---------------- Pub/Sub ----------------
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.on_imu, 100)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self._imu_log_bucket_last = None

        self.get_logger().info(
            f"READY | imu={self.imu_topic} motor={self.motor_topic} rate={self.rate_hz:.1f}Hz "
            f"motor_test={int(self.motor_test_enable)}(pwm={self.motor_test_pwm}) "
            f"calib={int(self.calib_enable)} fwd_target={self.fwd_target_pwm} turn_target={self.turn_target_pwm}"
        )

    # ---------------- Time helpers ----------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sec_since(self, t0: float) -> float:
        return self.now_s() - t0

    # ---------------- Motor publish helpers ----------------
    def publish_motor(self, left: int, right: int):
        if not rclpy.ok():
            return
        m = Int16MultiArray()
        m.data = [clampi(left, -1000, 1000), clampi(right, -1000, 1000)]
        self.pub_motor.publish(m)

    def stop(self):
        self.publish_motor(0, 0)

    # ---------------- IMU callback ----------------
    def on_imu(self, msg: Imu):
        q = msg.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        gz = float(msg.angular_velocity.z)  # assumed rad/s
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)

        # choose stamp source
        if self.use_header_stamp:
            # header stamp should be sec + nanosec(0..1e9)
            ns = int(msg.header.stamp.nanosec)
            sec = int(msg.header.stamp.sec)
            if 0 <= ns < 1_000_000_000 and sec > 0:
                stamp_s = float(sec) + float(ns) * 1e-9
            else:
                # fallback wall
                stamp_s = self.now_s()
        else:
            stamp_s = self.now_s()

        self.imu = ImuSample(stamp_s=stamp_s, yaw=yaw, gz=gz, ax=ax, ay=ay, az=az)
        self.have_imu = True

    # ---------------- IMU log ----------------
    def log_imu(self, prefix: str = "IMU"):
        bucket = int(self.now_s() * self.imu_log_hz)
        if bucket == self._imu_log_bucket_last:
            return
        self._imu_log_bucket_last = bucket

        self.get_logger().info(
            f"{prefix} | yaw={math.degrees(self.imu.yaw):.2f}deg "
            f"gz={math.degrees(self.imu.gz):.2f} dps "
            f"acc=({self.imu.ax:.2f},{self.imu.ay:.2f},{self.imu.az:.2f}) m/s^2"
        )

    # ---------------- Calibration routines ----------------
    def calib_detect_turn_sign_and_min_pwm(self) -> bool:
        if self.turn_sign_hint in (+1, -1):
            self.turn_sign = self.turn_sign_hint

        pwm = self.pwm_min_start
        detected = False

        while pwm <= self.pwm_max and rclpy.ok():
            self.stop()
            t_pause = self.now_s()
            while self.sec_since(t_pause) < self.calib_pause_s and rclpy.ok():
                time.sleep(0.01)

            yaw0 = self.imu.yaw
            gz_abs_sum = 0.0
            n = 0

            t0 = self.now_s()
            while self.sec_since(t0) < self.calib_spin_s and rclpy.ok():
                self.publish_motor(+pwm, -pwm)
                gz_abs_sum += abs(self.imu.gz)
                n += 1
                time.sleep(0.01)

            self.stop()
            yaw1 = self.imu.yaw
            yaw_delta = wrap_pi(yaw1 - yaw0)
            gz_abs_avg = (gz_abs_sum / max(1, n))

            moved_by_gyro = gz_abs_avg > self.gyro_thr
            moved_by_yaw = abs(yaw_delta) > self.yaw_move_thr
            moved = moved_by_gyro or moved_by_yaw

            self.get_logger().info(
                f"CALIB turn pwm={pwm} | gz_abs_avg={math.degrees(gz_abs_avg):.2f} dps "
                f"| yaw_delta={math.degrees(yaw_delta):.2f} deg | moved={int(moved)}"
            )

            if moved and not detected:
                detected = True
                self.min_turn_pwm = pwm

                if self.turn_sign_hint == 0:
                    self.turn_sign = +1 if yaw_delta >= 0.0 else -1

                self.get_logger().warn(
                    f"CALIB OK -> min_turn_pwm={self.min_turn_pwm} turn_sign={self.turn_sign:+d} "
                    f"(tank (+,-) makes yaw {'increase' if self.turn_sign > 0 else 'decrease'})"
                )
                return True

            pwm += self.pwm_step

        if self.min_turn_pwm is None:
            self.min_turn_pwm = max(450, self.pwm_min_start)
        if self.turn_sign == 0:
            self.turn_sign = +1

        self.get_logger().warn(
            f"CALIB fallback -> min_turn_pwm={self.min_turn_pwm} turn_sign={self.turn_sign:+d}"
        )
        return False

    def calib_forward_trim(self):
        base = clampi(self.fwd_target_pwm, 0, 1000)
        if base <= 0:
            return

        self.stop()
        time.sleep(self.calib_pause_s)

        yaw0 = self.imu.yaw
        t0 = self.now_s()
        while self.sec_since(t0) < self.calib_fwd_s and rclpy.ok():
            self.publish_motor(base, base)
            time.sleep(0.01)
        self.stop()

        yaw1 = self.imu.yaw
        drift = wrap_pi(yaw1 - yaw0)
        drift_deg = math.degrees(drift)
        new_trim = max(-0.20, min(0.20, drift_deg / 160.0))

        self.get_logger().warn(
            f"CALIB forward drift={drift_deg:.2f}deg over {self.calib_fwd_s:.2f}s -> "
            f"trim={new_trim:.3f} (prev={self.straight_trim:.3f})"
        )
        self.straight_trim = new_trim

    # ---------------- Control primitives ----------------
    def straight_hold(self, base_pwm: int, heading_ref: float):
        yaw = self.imu.yaw
        gz = self.imu.gz

        e = wrap_pi(yaw - heading_ref)
        delta = self.straight_kp * e + self.straight_kd * gz
        delta_i = clampi(int(delta), -self.straight_max_delta, self.straight_max_delta)

        trim_pwm = int(self.straight_trim * base_pwm)
        left = base_pwm - delta_i + trim_pwm
        right = base_pwm + delta_i - trim_pwm
        self.publish_motor(left, right)

    def turn_to_target(self, target_yaw: float, base_pwm: int) -> bool:
        yaw = self.imu.yaw
        gz = self.imu.gz
        e = wrap_pi(yaw - target_yaw)

        if abs(e) <= self.yaw_tol:
            self.stop()
            return True

        u = self.turn_kp * e + self.turn_kd * gz
        u_i = clampi(int(u), -base_pwm, base_pwm)

        left = -u_i
        right = +u_i

        minp = int(self.min_turn_pwm or self.pwm_min_start)
        if abs(left) < minp:
            left = minp if left >= 0 else -minp
        if abs(right) < minp:
            right = minp if right >= 0 else -minp

        self.publish_motor(left, right)
        return False

    # ---------------- Motor test FSM ----------------
    def motor_test_tick(self):
        if self.motor_step_i >= len(self.motor_steps):
            self.stop()
            self.get_logger().warn("MOTOR_TEST finished.")
            # next
            if self.calib_enable:
                self.state = State.CALIB
                self.get_logger().info("State -> CALIB")
            else:
                self.state = State.FWD1
                self.t_state0 = self.now_s()
                self.heading_ref = self.imu.yaw
                self.get_logger().info("State -> FWD1 (no calib)")
            return

        step = self.motor_steps[self.motor_step_i]

        if not self.motor_step_started:
            self.motor_step_started = True
            self.t_state0 = self.now_s()
            self.get_logger().warn(
                f"MOTOR_TEST step {self.motor_step_i+1}/{len(self.motor_steps)}: {step.name} "
                f"cmd=({step.left},{step.right}) dur={step.dur_s:.2f}s"
            )

        # command motors
        self.publish_motor(step.left, step.right)

        # log IMU during step
        self.log_imu(prefix=f"{step.name}")

        if self.sec_since(self.t_state0) >= step.dur_s:
            self.stop()
            self.motor_step_i += 1
            self.motor_step_started = False

    # ---------------- Main tick ----------------
    def tick(self):
        if self.state == State.WAIT_IMU:
            self.stop()
            if self.have_imu:
                self.get_logger().info("IMU OK -> start sequence")
                if self.motor_test_enable:
                    self.state = State.MOTOR_TEST
                    self.motor_step_i = 0
                    self.motor_step_started = False
                    self.get_logger().info("State -> MOTOR_TEST")
                else:
                    if self.calib_enable:
                        self.state = State.CALIB
                        self.get_logger().info("State -> CALIB")
                    else:
                        self.state = State.FWD1
                        self.t_state0 = self.now_s()
                        self.heading_ref = self.imu.yaw
                        self.get_logger().info("State -> FWD1 (no calib)")
            return

        if self.state == State.MOTOR_TEST:
            self.motor_test_tick()
            return

        if self.state == State.CALIB:
            self.stop()
            self.get_logger().warn("CALIB: robotu boş alanda yap (tank-turn + kısa forward).")
            ok = self.calib_detect_turn_sign_and_min_pwm()
            self.calib_forward_trim()

            self.state = State.FWD1
            self.t_state0 = self.now_s()
            self.heading_ref = self.imu.yaw

            self.get_logger().info(
                f"State -> FWD1 | min_turn_pwm={self.min_turn_pwm} turn_sign={self.turn_sign:+d} "
                f"trim={self.straight_trim:.3f} calib_ok={int(ok)}"
            )
            return

        if self.state == State.FWD1:
            self.log_imu(prefix="FWD1")
            if self.sec_since(self.t_state0) < self.t_fwd1:
                self.straight_hold(self.fwd_target_pwm, self.heading_ref)
            else:
                self.stop()
                self.state = State.TURN_RIGHT
                self.t_state0 = self.now_s()
                self.yaw_start = self.imu.yaw

                if self.turn_sign >= 0:
                    self.yaw_target = wrap_pi(self.yaw_start + self.turn_rad)
                else:
                    self.yaw_target = wrap_pi(self.yaw_start - self.turn_rad)

                self.get_logger().info(
                    f"State -> TURN_RIGHT | yaw_start={math.degrees(self.yaw_start):.1f}deg "
                    f"yaw_target={math.degrees(self.yaw_target):.1f}deg (turn_sign={self.turn_sign:+d})"
                )
            return

        if self.state == State.TURN_RIGHT:
            self.log_imu(prefix="TURN")
            done = self.turn_to_target(self.yaw_target, self.turn_target_pwm)
            if done:
                self.state = State.FWD2
                self.t_state0 = self.now_s()
                self.heading_ref = self.imu.yaw
                self.get_logger().info("State -> FWD2")
                return

            if self.sec_since(self.t_state0) > self.turn_timeout_s:
                self.stop()
                self.state = State.DONE
                self.get_logger().warn("TURN timeout -> DONE (safety)")
                return
            return

        if self.state == State.FWD2:
            self.log_imu(prefix="FWD2")
            if self.sec_since(self.t_state0) < self.t_fwd2:
                self.straight_hold(self.fwd_target_pwm, self.heading_ref)
            else:
                self.stop()
                self.state = State.DONE
                self.get_logger().info("State -> DONE")
            return

        if self.state == State.DONE:
            self.stop()

    def destroy_node(self):
        try:
            self.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ImuMotionTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()