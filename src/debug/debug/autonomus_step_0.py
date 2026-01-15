#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Imu


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AutonomousStep0(Node):
    """
    Sub:
      /camera_bottom/lane_info (std_msgs/String JSON)
      /imu (sensor_msgs/Imu)

    Pub:
      /motor_cmd (std_msgs/Int16MultiArray)  [left,right]  -1000..1000

    Davranış:
      WAIT_IMU  : IMU gelene kadar dur
      SEARCH    : düz ileri (yaw hold), lane görünene kadar
      ALIGN     : lane görünce yavaşla, dx ile ortala (yaw_ref sabit)
      DONE      : dur, 0.3s bas, node kapanır
    """

    def __init__(self):
        super().__init__("autonomous_step_0")

        # -------- params
        self.declare_parameter("lane_info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("motor_cmd_topic", "/motor_cmd")

        # motor komut (Int16MultiArray) ölçeği
        self.declare_parameter("search_base", 260)   # 0..1000
        self.declare_parameter("align_base", 140)    # 0..1000
        self.declare_parameter("max_steer", 260)     # 0..1000 (base civarı iyi)

        # lane detect
        self.declare_parameter("conf_enter", 0.60)
        self.declare_parameter("frames_confirm", 4)

        # dx toleransı
        self.declare_parameter("dx_tol", 0.05)       # (dx-0.5) toleransı
        self.declare_parameter("frames_settle", 6)

        # kontrol
        self.declare_parameter("k_yaw", 220.0)       # yaw_err(rad) -> steer units
        self.declare_parameter("k_dx", 520.0)        # dx_err -> steer units

        # topic timeout
        self.declare_parameter("lane_timeout_s", 0.7)

        # loop
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("shutdown_after_done_s", 0.3)

        self.lane_info_topic = self.get_parameter("lane_info_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.motor_cmd_topic = self.get_parameter("motor_cmd_topic").value

        self.search_base = int(self.get_parameter("search_base").value)
        self.align_base = int(self.get_parameter("align_base").value)
        self.max_steer = int(self.get_parameter("max_steer").value)

        self.conf_enter = float(self.get_parameter("conf_enter").value)
        self.frames_confirm = int(self.get_parameter("frames_confirm").value)

        self.dx_tol = float(self.get_parameter("dx_tol").value)
        self.frames_settle = int(self.get_parameter("frames_settle").value)

        self.k_yaw = float(self.get_parameter("k_yaw").value)
        self.k_dx = float(self.get_parameter("k_dx").value)

        self.lane_timeout_s = float(self.get_parameter("lane_timeout_s").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.shutdown_after_done_s = float(self.get_parameter("shutdown_after_done_s").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_lane = self.create_subscription(String, self.lane_info_topic, self.cb_lane, qos)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos)

        self.pub_motor = self.create_publisher(Int16MultiArray, self.motor_cmd_topic, 10)

        # -------- state
        self.state = "WAIT_IMU"
        self.yaw_ref = None
        self.yaw_now = None

        self.lane_visible = False
        self.lane_conf = 0.0
        self.dx = 0.5
        self.last_lane_stamp = 0.0

        self.confirm_cnt = 0
        self.settle_cnt = 0

        self.done_started_t = None

        self.timer = self.create_timer(1.0 / max(1.0, self.publish_rate_hz), self.tick)

        self.get_logger().info(
            f"[READY] state={self.state} pub={self.motor_cmd_topic} sub_lane={self.lane_info_topic} sub_imu={self.imu_topic}"
        )

    # ---------------- callbacks
    def cb_imu(self, msg: Imu):
        q = msg.orientation
        self.yaw_now = yaw_from_quat(q.x, q.y, q.z, q.w)

        if self.yaw_ref is None:
            self.yaw_ref = self.yaw_now
            self.state = "SEARCH"
            self.get_logger().info("[IMU_REF_SET] -> SEARCH")

    def cb_lane(self, msg: String):
        try:
            m = json.loads(msg.data)
        except Exception:
            return

        sel_vis = bool((m.get("lane_select") or {}).get("selected_visible", False))
        sel = m.get("selected") or {}
        self.lane_conf = float(sel.get("confidence", 0.0))
        self.dx = float(sel.get("center_offset_x_norm", 0.5))  # örneklerde 0.46-0.55

        self.lane_visible = sel_vis and (self.lane_conf >= self.conf_enter)
        self.last_lane_stamp = time.time()

    # ---------------- motor publish
    def publish_lr(self, left: int, right: int):
        left = int(max(-1000, min(1000, left)))
        right = int(max(-1000, min(1000, right)))
        msg = Int16MultiArray()
        msg.data = [left, right]
        self.pub_motor.publish(msg)

    def stop(self):
        self.publish_lr(0, 0)

    # ---------------- loop
    def tick(self):
        # lane timeout -> görünmez say
        if time.time() - self.last_lane_stamp > self.lane_timeout_s:
            self.lane_visible = False
            self.lane_conf = 0.0

        if self.state == "WAIT_IMU":
            self.stop()
            return

        if self.yaw_now is None or self.yaw_ref is None:
            self.stop()
            return

        yaw_err = wrap_pi(self.yaw_ref - self.yaw_now)

        # dx hatası: merkez 0.5 kabulü
        # Eğer sende dx zaten -1..+1 offset ise şu satırı: dx_err = self.dx yap.
        dx_err = (self.dx - 0.5)

        # ---------- SEARCH (düz ileri + yaw hold)
        if self.state == "SEARCH":
            steer = (self.k_yaw * yaw_err)
            steer = clamp(steer, -self.max_steer, self.max_steer)

            base = self.search_base
            left = base - steer
            right = base + steer
            self.publish_lr(int(left), int(right))

            if self.lane_visible:
                self.confirm_cnt += 1
            else:
                self.confirm_cnt = 0

            if self.confirm_cnt >= self.frames_confirm:
                self.state = "ALIGN"
                self.confirm_cnt = 0
                self.settle_cnt = 0
                self.get_logger().info("[LANE_CONFIRMED] -> ALIGN")

            return

        # ---------- ALIGN (yavaş + dx ile ortala, yaw_ref sabit)
        if self.state == "ALIGN":
            steer = (self.k_dx * dx_err) + (self.k_yaw * yaw_err)
            steer = clamp(steer, -self.max_steer, self.max_steer)

            base = self.align_base
            left = base - steer
            right = base + steer
            self.publish_lr(int(left), int(right))

            if self.lane_visible and (abs(dx_err) <= self.dx_tol):
                self.settle_cnt += 1
            else:
                self.settle_cnt = 0

            if self.settle_cnt >= self.frames_settle:
                self.state = "DONE"
                self.done_started_t = time.time()
                self.get_logger().info("[CENTERED] -> DONE (stop & shutdown)")
            return

        # ---------- DONE (dur + kapat)
        if self.state == "DONE":
            self.stop()
            if self.done_started_t is not None:
                if (time.time() - self.done_started_t) >= self.shutdown_after_done_s:
                    try:
                        self.timer.cancel()
                    except Exception:
                        pass
                    self.destroy_node()
                    rclpy.shutdown()
            return


def main():
    rclpy.init()
    node = AutonomousStep0()
    rclpy.spin(node)


if __name__ == "__main__":
    main()