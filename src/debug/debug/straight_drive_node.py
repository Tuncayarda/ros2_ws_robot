#!/usr/bin/env python3
import math
import time
from enum import Enum
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from std_msgs.msg import Int16MultiArray


def clampi(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def wrap_pi(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw around world Z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ImuSample:
    stamp_s: float = 0.0
    yaw: float = 0.0
    gz: float = 0.0  # rad/s (angular_velocity.z)


class Phase(Enum):
    WAIT_IMU = 0
    SETTLE = 1
    STEER_TEST = 2
    DRIVE = 3
    DONE = 4


class StraightDriveNode(Node):
    def __init__(self):
        super().__init__("straight_drive_node")

        # topics
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_topic", "/motor_cmd")

        # timing
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("settle_s", 0.4)
        self.declare_parameter("drive_time_s", 3.0)
        self.declare_parameter("ramp_s", 1.0)

        # base speed
        self.declare_parameter("base_target_pwm", 650)
        self.declare_parameter("base_start_pwm", 350)

        # controller
        self.declare_parameter("kp", 240.0)          # PWM per rad
        self.declare_parameter("kd", 25.0)           # PWM per (rad/s)
        self.declare_parameter("max_delta", 250)     # clamp delta

        # steer sign
        self.declare_parameter("sign_hint", 0)       # 0=auto, +1 or -1 force
        self.declare_parameter("steer_test_pwm", 120)
        self.declare_parameter("steer_test_s", 0.25)

        # yaw reference behavior
        self.declare_parameter("ref_mode", "hold")   # "hold" or "update" (hold recommended)

        # ---------------- read params ----------------
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)

        self.rate_hz = max(5.0, float(self.get_parameter("rate_hz").value))
        self.settle_s = float(self.get_parameter("settle_s").value)
        self.drive_time_s = float(self.get_parameter("drive_time_s").value)
        self.ramp_s = max(0.05, float(self.get_parameter("ramp_s").value))

        self.base_target = clampi(int(self.get_parameter("base_target_pwm").value), 0, 1000)
        self.base_start = clampi(int(self.get_parameter("base_start_pwm").value), 0, 1000)

        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)
        self.max_delta = int(self.get_parameter("max_delta").value)

        self.sign_hint = int(self.get_parameter("sign_hint").value)
        self.test_pwm = clampi(int(self.get_parameter("steer_test_pwm").value), 0, 500)
        self.test_s = float(self.get_parameter("steer_test_s").value)

        self.ref_mode = str(self.get_parameter("ref_mode").value).strip().lower()

        # ---------------- state ----------------
        self.phase = Phase.WAIT_IMU
        self.have_imu = False
        self.imu = ImuSample()

        self.t0 = self.now_s()
        self.yaw_ref = 0.0

        # steer_sign meaning:
        # we compute "delta_cmd" from (kp*e + kd*gz).
        # Then we apply:
        #   left  = base - steer_sign*delta_cmd
        #   right = base + steer_sign*delta_cmd
        # So steer_sign must be chosen so that positive e reduces over time.
        self.steer_sign = 0
        if self.sign_hint in (+1, -1):
            self.steer_sign = self.sign_hint

        self._dbg_last_bucket = None

        # pub/sub
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.on_imu, 100)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(
            f"straight_drive READY | imu={self.imu_topic} motor={self.motor_topic} "
            f"rate={self.rate_hz:.1f}Hz drive_time={self.drive_time_s:.2f}s ramp={self.ramp_s:.2f}s "
            f"base_start={self.base_start} base_target={self.base_target} kp={self.kp} kd={self.kd} "
            f"max_delta={self.max_delta} sign_hint={self.sign_hint} ref_mode={self.ref_mode}"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sec_since(self, t: float) -> float:
        return self.now_s() - t

    def publish_motor(self, left: int, right: int):
        m = Int16MultiArray()
        m.data = [clampi(left, -1000, 1000), clampi(right, -1000, 1000)]
        self.pub_motor.publish(m)

    def stop(self):
        self.publish_motor(0, 0)

    def on_imu(self, msg: Imu):
        q = msg.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        gz = float(msg.angular_velocity.z)  # rad/s
        self.imu = ImuSample(stamp_s=self.now_s(), yaw=yaw, gz=gz)
        self.have_imu = True

    def dbg(self, text: str):
        # ~5Hz log
        bucket = int(self.now_s() * 5.0)
        if bucket != self._dbg_last_bucket:
            self._dbg_last_bucket = bucket
            self.get_logger().info(text)

    def steer_test_auto_sign(self):
        """
        Apply a short +delta and see yaw change.
        If (+delta) makes yaw INCREASE -> steer_sign = -1 (so we flip)
        If (+delta) makes yaw DECREASE -> steer_sign = +1
        """
        if self.steer_sign in (+1, -1):
            self.get_logger().info(f"STEER_TEST skipped (forced sign={self.steer_sign:+d})")
            return

        self.stop()
        time.sleep(0.15)

        yaw0 = self.imu.yaw

        # apply +delta briefly: left slower, right faster (candidate)
        # candidate steer_sign = +1 means:
        # left=base-delta, right=base+delta
        base = max(self.base_start, 350)
        delta = self.test_pwm

        t = self.now_s()
        while self.sec_since(t) < self.test_s and rclpy.ok():
            self.publish_motor(base - delta, base + delta)
            time.sleep(0.01)
        self.stop()
        time.sleep(0.10)

        yaw1 = self.imu.yaw
        dyaw = wrap_pi(yaw1 - yaw0)
        dyaw_deg = math.degrees(dyaw)

        # If yaw increased with (+delta), then (+delta) turns the robot "left" in yaw sense.
        # For our controller, we want positive error to be corrected by applying delta that reduces error.
        # Easiest: choose steer_sign so that when error e is positive, produced delta makes yaw go back.
        #
        # Here we just set sign so that our delta_cmd is applied in the OPPOSITE direction of this test:
        # yaw increased -> set steer_sign = -1, yaw decreased -> +1
        self.steer_sign = -1 if dyaw >= 0.0 else +1

        self.get_logger().warn(
            f"STEER_TEST: applied (+delta) -> yaw_delta={dyaw_deg:+.2f}deg "
            f"=> steer_sign={self.steer_sign:+d}  "
            f"(controller will apply sign*delta)"
        )

    def base_ramp(self, t_in_drive: float) -> int:
        if self.base_target <= 0:
            return 0
        if t_in_drive <= 0.0:
            return self.base_start
        if t_in_drive >= self.ramp_s:
            return self.base_target
        a = t_in_drive / self.ramp_s
        return int(self.base_start + a * (self.base_target - self.base_start))

    def tick(self):
        if not self.have_imu:
            self.stop()
            if self.phase != Phase.WAIT_IMU:
                self.phase = Phase.WAIT_IMU
            return

        if self.phase == Phase.WAIT_IMU:
            self.stop()
            self.t0 = self.now_s()
            self.phase = Phase.SETTLE
            self.get_logger().info("Phase -> SETTLE (waiting IMU stable)")
            return

        if self.phase == Phase.SETTLE:
            self.stop()
            if self.sec_since(self.t0) >= self.settle_s:
                self.phase = Phase.STEER_TEST
                self.get_logger().info("Phase -> STEER_TEST")
            return

        if self.phase == Phase.STEER_TEST:
            # set yaw_ref from current
            self.yaw_ref = self.imu.yaw
            self.get_logger().info(f"STEER_TEST start | yaw_ref={math.degrees(self.yaw_ref):.2f}deg")
            self.steer_test_auto_sign()
            self.phase = Phase.DRIVE
            self.t0 = self.now_s()
            self.get_logger().info(f"Phase -> DRIVE | steer_sign={self.steer_sign:+d}")
            return

        if self.phase == Phase.DRIVE:
            t = self.sec_since(self.t0)
            if t >= self.drive_time_s:
                self.stop()
                self.phase = Phase.DONE
                self.get_logger().info("Phase -> DONE (stopped)")
                return

            # optional: update reference slowly (usually keep hold)
            if self.ref_mode == "update":
                # tiny low-pass towards current yaw (prevents drift accumulation)
                self.yaw_ref = wrap_pi(self.yaw_ref + 0.02 * wrap_pi(self.imu.yaw - self.yaw_ref))

            yaw = self.imu.yaw
            gz = self.imu.gz
            e = wrap_pi(yaw - self.yaw_ref)  # rad (positive means yaw > ref)

            # PD
            delta_cmd = self.kp * e + self.kd * gz
            delta = clampi(int(delta_cmd), -self.max_delta, self.max_delta)

            base = self.base_ramp(t)

            # apply correction with sign
            s = self.steer_sign if self.steer_sign in (+1, -1) else +1
            left = base - s * delta
            right = base + s * delta

            self.publish_motor(left, right)

            self.dbg(
                f"yaw={math.degrees(yaw):.2f}deg ref={math.degrees(self.yaw_ref):.2f}deg "
                f"e={math.degrees(e):+.2f}deg gz={math.degrees(gz):+.2f}dps "
                f"base={base} delta={delta} sign={s:+d} L={left} R={right}"
            )
            return

        if self.phase == Phase.DONE:
            self.stop()

    def destroy_node(self):
        try:
            self.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = StraightDriveNode()
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