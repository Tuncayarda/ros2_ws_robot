#!/usr/bin/env python3
import time
import math
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Imu


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw in radians [-pi, +pi]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def wrap_deg(a: float) -> float:
    # [-180, 180)
    while a >= 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class ImuTurnPidTeleop(Node):
    """
    UP hold: forward
    LEFT/RIGHT: target = yaw_corr +/- turn_step_deg, PID ile hedefe kilitlen
    SPACE: stop
    Q/ESC: quit

    Tek knoblar:
      - yaw_sign: IMU yaw tersse -1 yap
      - turn_motor_sign: motor dönüş yönü tersse -1 yap
    """

    def __init__(self):
        super().__init__("imu_turn_pid_teleop")

        # topics
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_topic", "/motor_cmd")

        # rates
        self.declare_parameter("control_hz", 80.0)
        self.declare_parameter("imu_timeout_s", 0.35)

        # forward
        self.declare_parameter("fwd_pwm", 220.0)
        self.declare_parameter("slew_per_tick", 80)

        # turn
        self.declare_parameter("turn_step_deg", 90.0)
        self.declare_parameter("turn_tol_deg", 1.0)
        self.declare_parameter("settle_s", 0.20)

        # PID (error in degrees)
        self.declare_parameter("kp", 7.0)
        self.declare_parameter("ki", 0.25)
        self.declare_parameter("kd", 1.3)
        self.declare_parameter("i_clamp_pwm", 220.0)

        # pwm limits
        self.declare_parameter("min_turn_pwm", 250.0)   # başlangıç 250 dedin
        self.declare_parameter("max_turn_pwm", 750.0)
        self.declare_parameter("fine_band_deg", 10.0)
        self.declare_parameter("fine_min_pwm", 120.0)
        self.declare_parameter("fine_max_pwm", 320.0)

        # two simple knobs
        self.declare_parameter("yaw_sign", 1.0)         # IMU yaw tersse -1.0
        self.declare_parameter("turn_motor_sign", 1.0)  # motor dönüş tersse -1.0

        # optional trim/scale
        self.declare_parameter("left_trim", 0.0)
        self.declare_parameter("right_trim", 0.0)
        self.declare_parameter("left_scale", 1.0)
        self.declare_parameter("right_scale", 1.0)

        # read
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.imu_timeout_s = float(self.get_parameter("imu_timeout_s").value)

        self.fwd_pwm = float(self.get_parameter("fwd_pwm").value)
        self.slew_per_tick = int(self.get_parameter("slew_per_tick").value)

        self.turn_step_deg = float(self.get_parameter("turn_step_deg").value)
        self.turn_tol_deg = float(self.get_parameter("turn_tol_deg").value)
        self.settle_s = float(self.get_parameter("settle_s").value)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.i_clamp_pwm = float(self.get_parameter("i_clamp_pwm").value)

        self.min_turn_pwm = float(self.get_parameter("min_turn_pwm").value)
        self.max_turn_pwm = float(self.get_parameter("max_turn_pwm").value)
        self.fine_band_deg = float(self.get_parameter("fine_band_deg").value)
        self.fine_min_pwm = float(self.get_parameter("fine_min_pwm").value)
        self.fine_max_pwm = float(self.get_parameter("fine_max_pwm").value)

        self.yaw_sign = float(self.get_parameter("yaw_sign").value)
        self.turn_motor_sign = float(self.get_parameter("turn_motor_sign").value)

        self.left_trim = float(self.get_parameter("left_trim").value)
        self.right_trim = float(self.get_parameter("right_trim").value)
        self.left_scale = float(self.get_parameter("left_scale").value)
        self.right_scale = float(self.get_parameter("right_scale").value)

        # ROS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos)
        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_topic, 10)

        # state
        self.last_imu_t = 0.0
        self.yaw_raw_deg: Optional[float] = None
        self.yaw_corr_deg: Optional[float] = None

        self.cmd_l = 0
        self.cmd_r = 0

        # keyboard intents
        self._lock = threading.Lock()
        self._want_fwd = False
        self._want_stop = False
        self._want_quit = False
        self._want_turn_left = False
        self._want_turn_right = False

        # turn state
        self.turning = False
        self.target_corr_deg = 0.0
        self._in_tol_since: Optional[float] = None

        # PID state
        self._e_prev = 0.0
        self._i_state = 0.0

        # timers
        self._t_last = time.time()
        dt = 1.0 / max(1.0, self.control_hz)
        self.timer = self.create_timer(dt, self.tick)

        # keyboard thread
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

        self.get_logger().info("READY: target = yaw + step, PID lock")
        self.get_logger().info("Fix knobs: yaw_sign (-1 if IMU yaw inverted), turn_motor_sign (-1 if motor turn inverted)")

    def cb_imu(self, msg: Imu):
        self.last_imu_t = time.time()
        q = msg.orientation
        yaw = wrap_deg(rad2deg(yaw_from_quat(q.x, q.y, q.z, q.w)))
        self.yaw_raw_deg = yaw
        self.yaw_corr_deg = wrap_deg(self.yaw_sign * yaw)

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

    def slew(self, target: int, current: int) -> int:
        d = target - current
        if d > self.slew_per_tick:
            d = self.slew_per_tick
        elif d < -self.slew_per_tick:
            d = -self.slew_per_tick
        return current + d

    def _pid_reset(self, e0: float):
        self._e_prev = e0
        self._i_state = 0.0
        self._in_tol_since = None

    def start_turn(self, step_deg: float):
        if self.yaw_corr_deg is None:
            return
        self.turning = True
        self.target_corr_deg = wrap_deg(self.yaw_corr_deg + step_deg)
        e0 = wrap_deg(self.target_corr_deg - self.yaw_corr_deg)
        self._pid_reset(e0)
        self.get_logger().info(f"TURN start yaw_corr={self.yaw_corr_deg:+.2f} target_corr={self.target_corr_deg:+.2f}")

    def _turn_pid(self, dt: float):
        assert self.yaw_corr_deg is not None
        e = wrap_deg(self.target_corr_deg - self.yaw_corr_deg)  # shortest error [-180,180)

        # settle
        if abs(e) <= self.turn_tol_deg:
            if self._in_tol_since is None:
                self._in_tol_since = time.time()
            elif (time.time() - self._in_tol_since) >= self.settle_s:
                self.turning = False
                return 0.0, 0.0, True, e
        else:
            self._in_tol_since = None

        # PID
        de = (e - self._e_prev) / max(1e-3, dt)
        self._e_prev = e

        self._i_state += e * dt
        i_pwm = clamp(self.ki * self._i_state, -self.i_clamp_pwm, self.i_clamp_pwm)

        u = self.kp * e + i_pwm + self.kd * de  # signed pwm command (positive means "turn towards +error")

        # choose pwm bounds (fine band)
        if abs(e) <= self.fine_band_deg:
            minp, maxp = self.fine_min_pwm, self.fine_max_pwm
        else:
            minp, maxp = self.min_turn_pwm, self.max_turn_pwm

        s = 1.0 if u >= 0.0 else -1.0
        pwm = abs(u)
        pwm = max(minp, pwm)
        pwm = min(maxp, pwm)
        pwm *= s

        # motor mapping: positive pwm should turn one direction; fix with turn_motor_sign
        pwm *= self.turn_motor_sign

        left = pwm
        right = -pwm
        return left, right, False, e

    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self._t_last)
        self._t_last = now

        if (now - self.last_imu_t) > self.imu_timeout_s or self.yaw_corr_deg is None:
            self.cmd_l = self.slew(0, self.cmd_l)
            self.cmd_r = self.slew(0, self.cmd_r)
            self.publish_motor(self.cmd_l, self.cmd_r)
            return

        with self._lock:
            want_fwd = self._want_fwd
            want_stop = self._want_stop
            want_quit = self._want_quit
            want_L = self._want_turn_left
            want_R = self._want_turn_right
            self._want_turn_left = False
            self._want_turn_right = False

        if want_quit:
            self.publish_motor(0, 0)
            rclpy.shutdown()
            return

        if want_stop:
            self.turning = False
            self.cmd_l = self.slew(0, self.cmd_l)
            self.cmd_r = self.slew(0, self.cmd_r)
            self.publish_motor(self.cmd_l, self.cmd_r)
            with self._lock:
                self._want_stop = False
            return

        # start turn (priority)
        if not self.turning:
            if want_R:
                self.start_turn(+self.turn_step_deg)
            elif want_L:
                self.start_turn(-self.turn_step_deg)

        if self.turning:
            left, right, done, e = self._turn_pid(dt)
            if done:
                self.cmd_l = self.slew(0, self.cmd_l)
                self.cmd_r = self.slew(0, self.cmd_r)
                self.publish_motor(self.cmd_l, self.cmd_r)
                self.get_logger().info(f"TURN done yaw_corr={self.yaw_corr_deg:+.2f} err={e:+.2f}")
                return

            l_cmd, r_cmd = self.motor_map(left, right)
            self.cmd_l = self.slew(l_cmd, self.cmd_l)
            self.cmd_r = self.slew(r_cmd, self.cmd_r)
            self.publish_motor(self.cmd_l, self.cmd_r)
            return

        # forward hold
        base = self.fwd_pwm if want_fwd else 0.0
        l_cmd, r_cmd = self.motor_map(base, base)
        self.cmd_l = self.slew(l_cmd, self.cmd_l)
        self.cmd_r = self.slew(r_cmd, self.cmd_r)
        self.publish_motor(self.cmd_l, self.cmd_r)

    def _keyboard_loop(self):
        try:
            import curses
        except Exception as e:
            self.get_logger().error(f"curses import failed: {e}")
            return

        def loop(stdscr):
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            stdscr.nodelay(True)

            stdscr.clear()
            stdscr.addstr(0, 0, "IMU TURN PID TELEOP (target = yaw + 90)")
            stdscr.addstr(1, 0, "UP hold=forward | LEFT/RIGHT=turn | SPACE=stop | Q/ESC=quit")
            stdscr.addstr(2, 0, "Fix: yaw_sign=-1 if IMU inverted, turn_motor_sign=-1 if motor turn inverted")
            stdscr.refresh()

            last_up_t = 0.0
            release_s = 0.12
            poll_dt = 0.02

            while rclpy.ok():
                now = time.time()
                ch = stdscr.getch()

                if ch != -1:
                    if ch == curses.KEY_UP:
                        last_up_t = now
                        with self._lock:
                            self._want_fwd = True
                    elif ch == curses.KEY_LEFT:
                        with self._lock:
                            self._want_fwd = False
                            self._want_turn_left = True
                    elif ch == curses.KEY_RIGHT:
                        with self._lock:
                            self._want_fwd = False
                            self._want_turn_right = True
                    elif ch == ord(' '):
                        with self._lock:
                            self._want_stop = True
                    elif ch in (ord('q'), ord('Q'), 27):
                        with self._lock:
                            self._want_quit = True
                        break

                if (now - last_up_t) > release_s:
                    with self._lock:
                        self._want_fwd = False

                try:
                    yr = self.yaw_raw_deg
                    yc = self.yaw_corr_deg
                    stdscr.addstr(4, 0, f"yaw_raw : {yr:+8.2f}      " if yr is not None else "yaw_raw : (none)      ")
                    stdscr.addstr(5, 0, f"yaw_corr: {yc:+8.2f}      " if yc is not None else "yaw_corr: (none)      ")
                    if self.turning and yc is not None:
                        e = wrap_deg(self.target_corr_deg - yc)
                        stdscr.addstr(6, 0, f"turning: True target:{self.target_corr_deg:+8.2f} err:{e:+7.2f}      ")
                    else:
                        stdscr.addstr(6, 0, f"turning: False                                      ")
                    stdscr.refresh()
                except Exception:
                    pass

                time.sleep(poll_dt)

        try:
            import curses
            curses.wrapper(loop)
        except Exception as e:
            try:
                self.publish_motor(0, 0)
            except Exception:
                pass
            self.get_logger().error(f"Keyboard loop error: {e}")


def main():
    rclpy.init()
    node = ImuTurnPidTeleop()
    rclpy.spin(node)
    try:
        node.publish_motor(0, 0)
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()