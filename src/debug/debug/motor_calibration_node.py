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


@dataclass
class ImuWin:
    # running window for gz
    sum_abs_gz: float = 0.0
    n: int = 0

    def reset(self):
        self.sum_abs_gz = 0.0
        self.n = 0

    def add(self, gz_rad_s: float):
        self.sum_abs_gz += abs(float(gz_rad_s))
        self.n += 1

    def avg_abs(self) -> float:
        if self.n <= 0:
            return 0.0
        return self.sum_abs_gz / self.n


class State(Enum):
    WAIT_IMU = 0
    RIGHT_TEST = 1
    LEFT_TEST = 2
    DONE = 3


class MotorCalibrationNode(Node):
    def __init__(self):
        super().__init__("motor_calibration_node")

        # -------- params --------
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_topic", "/motor_cmd")
        self.declare_parameter("rate_hz", 50.0)

        # sweep params
        self.declare_parameter("start_pwm", 250)
        self.declare_parameter("step_pwm", 25)
        self.declare_parameter("max_pwm", 900)

        # timing
        self.declare_parameter("settle_s", 0.25)   # after stop, wait a bit
        self.declare_parameter("test_s", 0.80)     # how long to apply a pwm step
        self.declare_parameter("cooldown_s", 0.35) # between pwm steps

        # detection
        self.declare_parameter("gyro_thr_dps", 8.0)     # movement threshold (deg/s) on |gz|
        self.declare_parameter("require_consecutive", 2) # how many consecutive passes needed
        self.declare_parameter("log_each_step", True)

        # -------- read params --------
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)
        self.rate_hz = max(5.0, float(self.get_parameter("rate_hz").value))

        self.start_pwm = clampi(int(self.get_parameter("start_pwm").value), 0, 1000)
        self.step_pwm = max(1, int(self.get_parameter("step_pwm").value))
        self.max_pwm = clampi(int(self.get_parameter("max_pwm").value), 0, 1000)

        self.settle_s = float(self.get_parameter("settle_s").value)
        self.test_s = float(self.get_parameter("test_s").value)
        self.cooldown_s = float(self.get_parameter("cooldown_s").value)

        self.gyro_thr = math.radians(float(self.get_parameter("gyro_thr_dps").value))  # rad/s
        self.require_consecutive = max(1, int(self.get_parameter("require_consecutive").value))
        self.log_each_step = bool(self.get_parameter("log_each_step").value)

        # -------- state --------
        self.state = State.WAIT_IMU
        self.have_imu = False

        self.gz_win = ImuWin()
        self._testing = False
        self._t_step0 = 0.0
        self._t_after_stop0 = 0.0
        self._pwm = self.start_pwm
        self._consecutive_ok = 0

        self.min_pwm_right = None
        self.min_pwm_left = None

        # -------- ROS I/O --------
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.on_imu, 100)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(
            f"motor_calibration_node READY | imu={self.imu_topic} motor={self.motor_topic} "
            f"sweep: start={self.start_pwm} step={self.step_pwm} max={self.max_pwm} "
            f"test_s={self.test_s:.2f} settle_s={self.settle_s:.2f} cooldown_s={self.cooldown_s:.2f} "
            f"gyro_thr={math.degrees(self.gyro_thr):.1f} dps consecutive={self.require_consecutive}"
        )

    # -------- helpers --------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sec_since(self, t0: float) -> float:
        return self.now_s() - t0

    def publish_motor(self, left: int, right: int):
        m = Int16MultiArray()
        m.data = [clampi(left, -1000, 1000), clampi(right, -1000, 1000)]
        self.pub_motor.publish(m)

    def stop(self):
        self.publish_motor(0, 0)

    # -------- IMU cb --------
    def on_imu(self, msg: Imu):
        self.have_imu = True
        gz = float(msg.angular_velocity.z)
        # only accumulate while a test step is running
        if self._testing:
            self.gz_win.add(gz)

    # -------- test step core --------
    def _start_step(self, motor: str):
        # motor: "RIGHT" or "LEFT"
        self.gz_win.reset()
        self._testing = True
        self._t_step0 = self.now_s()
        self._consecutive_ok = max(0, self._consecutive_ok)

        if motor == "RIGHT":
            self.publish_motor(0, +self._pwm)
        else:
            self.publish_motor(+self._pwm, 0)

        if self.log_each_step:
            self.get_logger().info(f"{motor} step start | pwm={self._pwm}")

    def _finish_step_and_eval(self, motor: str) -> bool:
        # returns True if detected movement for this step
        self._testing = False
        self.stop()

        gz_avg_abs = self.gz_win.avg_abs()
        moved = gz_avg_abs >= self.gyro_thr

        if self.log_each_step:
            self.get_logger().info(
                f"{motor} step end | pwm={self._pwm} | avg|gz|={math.degrees(gz_avg_abs):.2f} dps "
                f"thr={math.degrees(self.gyro_thr):.2f} dps -> moved={int(moved)}"
            )

        return moved

    # -------- main FSM tick --------
    def tick(self):
        if not rclpy.ok():
            return

        if self.state == State.WAIT_IMU:
            self.stop()
            if self.have_imu:
                self.get_logger().warn(
                    "IMU OK. Calibration starting: first RIGHT motor only, then LEFT motor only. "
                    "Robot yerde kalsın, boş alanda yap."
                )
                self.state = State.RIGHT_TEST
                self._pwm = self.start_pwm
                self._consecutive_ok = 0
                self._t_after_stop0 = self.now_s()
            return

        if self.state == State.RIGHT_TEST:
            done = self._run_motor_sweep("RIGHT")
            if done:
                self.state = State.LEFT_TEST
                self._pwm = self.start_pwm
                self._consecutive_ok = 0
                self._t_after_stop0 = self.now_s()
            return

        if self.state == State.LEFT_TEST:
            done = self._run_motor_sweep("LEFT")
            if done:
                self.state = State.DONE
                self.stop()
                self._print_result()
            return

        if self.state == State.DONE:
            self.stop()
            return

    def _run_motor_sweep(self, motor: str) -> bool:
        """
        returns True when that motor's min pwm found (or max reached -> fallback)
        """
        # enforce settle time after stopping
        if self._testing:
            # step running
            if self.sec_since(self._t_step0) >= self.test_s:
                moved = self._finish_step_and_eval(motor)

                if moved:
                    self._consecutive_ok += 1
                else:
                    self._consecutive_ok = 0

                # need cooldown after each step end
                self._t_after_stop0 = self.now_s()

                # if enough consecutive ok -> accept
                if self._consecutive_ok >= self.require_consecutive:
                    if motor == "RIGHT":
                        self.min_pwm_right = self._pwm
                    else:
                        self.min_pwm_left = self._pwm

                    self.get_logger().warn(
                        f"{motor} MIN PWM FOUND = {self._pwm} "
                        f"(require_consecutive={self.require_consecutive})"
                    )
                    return True

                # else increase pwm and continue
                self._pwm += self.step_pwm
                if self._pwm > self.max_pwm:
                    # fallback
                    fallback = self.max_pwm
                    if motor == "RIGHT":
                        self.min_pwm_right = fallback
                    else:
                        self.min_pwm_left = fallback
                    self.get_logger().warn(
                        f"{motor} min PWM not detected until max. Fallback={fallback}. "
                        "Eşiği düşür / test_s artır / robotu daha serbest bırak."
                    )
                    return True
            return False

        # not testing => wait settle+cooldown then start next step
        wait_needed = max(self.settle_s, self.cooldown_s)
        if self.sec_since(self._t_after_stop0) < wait_needed:
            return False

        # start next step with current pwm
        if self._pwm > self.max_pwm:
            # should not happen, but guard
            if motor == "RIGHT":
                self.min_pwm_right = self.max_pwm
            else:
                self.min_pwm_left = self.max_pwm
            return True

        self._start_step(motor)
        return False

    def _print_result(self):
        r = self.min_pwm_right if self.min_pwm_right is not None else -1
        l = self.min_pwm_left if self.min_pwm_left is not None else -1

        # suggestion: use the higher one as "safe minimum"
        safe_min = max(r, l)

        self.get_logger().warn("======== MOTOR CALIBRATION RESULT ========")
        self.get_logger().warn(f"min_pwm_right = {r}")
        self.get_logger().warn(f"min_pwm_left  = {l}")
        self.get_logger().warn(f"safe_min_pwm  = {safe_min}  (ikisini de garanti döndürmek için)")
        if r != -1 and l != -1:
            if r > l:
                self.get_logger().warn(
                    f"Sağ motor daha zor dönüyor. Eşitlemek için sağa yakın değer kullan "
                    f"ya da düşük hızlarda left/right scale uygula (örn right_scale={l/r:.3f})."
                )
            elif l > r:
                self.get_logger().warn(
                    f"Sol motor daha zor dönüyor. Eşitlemek için sola yakın değer kullan "
                    f"ya da düşük hızlarda left/right scale uygula (örn left_scale={r/l:.3f})."
                )
            else:
                self.get_logger().warn("İki motor minimumu aynı görünüyor.")
        self.get_logger().warn("=========================================")


def main():
    rclpy.init()
    node = MotorCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()