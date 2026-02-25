#!/usr/bin/env python3
"""
lane_controller_node
────────────────────
lane_detector'dan gelen algılama verisine göre motorları sürer.
Kesişim yönetimi, dönüş komutu ve durum makinesi burada.

Subscribe:
  /lane/detection       (std_msgs/String)  JSON detection
  /lane/turn_cmd        (std_msgs/String)  "left" veya "right"

Publish:
  /motor_cmd            (std_msgs/Int16MultiArray)  [left, right]
  /lane/status          (std_msgs/String)  FOLLOWING / WAITING / TURNING
"""
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class LaneControllerNode(Node):
    def __init__(self):
        super().__init__("lane_controller_node")

        # ── Topics ──
        self.declare_parameter("detection_topic", "/lane/detection")
        self.declare_parameter("motor_topic", "/motor_cmd")
        self.declare_parameter("turn_cmd_topic", "/lane/turn_cmd")
        self.declare_parameter("status_topic", "/lane/status")

        # ── Motor kontrol ──
        self.declare_parameter("base_speed", 140)
        self.declare_parameter("kp", 200.0)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 40.0)
        self.declare_parameter("max_speed", 300)
        self.declare_parameter("max_turn", 140)
        self.declare_parameter("err_deadband_px", 6)
        self.declare_parameter("turn_smoothing", 0.3)
        self.declare_parameter("invert_steering", True)

        # ── Kamera offset (piksel) ──
        # Kamera ortada değilse: pozitif = kamera sağa kayık → hedef sola kayar
        self.declare_parameter("camera_offset_px", 0)

        # ── Kesişim ──
        self.declare_parameter("intersection_confirm_frames", 4)
        self.declare_parameter("reverse_speed", 120)        # geri hız
        self.declare_parameter("reverse_duration", 0.6)     # geri süresi (saniye)

        # ── Dönüş manevrası ──
        # Artık sert yerinde dönüş yerine ileri-sağ/ileri-sol yapıyor
        self.declare_parameter("turn_speed_inner", 60)      # iç teker (yavaş)
        self.declare_parameter("turn_speed_outer", 220)     # dış teker (hızlı)
        self.declare_parameter("turn_duration", 1.5)
        self.declare_parameter("forward_after_turn", 0.4)

        # ── IO ──
        det_topic = self.get_parameter("detection_topic").value
        motor_topic = self.get_parameter("motor_topic").value
        turn_cmd_topic = self.get_parameter("turn_cmd_topic").value
        status_topic = self.get_parameter("status_topic").value

        self.motor_pub = self.create_publisher(Int16MultiArray, motor_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.create_subscription(String, det_topic, self.on_detection, 10)
        self.create_subscription(String, turn_cmd_topic, self.on_turn_cmd, 10)

        # ── Durum makinesi ──
        # FOLLOWING → REVERSING → WAITING → TURNING → FOLLOWING
        self.state = "FOLLOWING"
        self.intersection_counter = 0
        self.pending_turn = None
        self.turn_start_time = None
        self.turn_phase = None
        self.prev_turn = 0
        self.integral = 0.0
        self.prev_err = 0.0
        self.last_time = time.time()
        self.reverse_start_time = None

        self.get_logger().info(
            f"lane_controller READY | det={det_topic} motor={motor_topic} "
            f"turn_cmd={turn_cmd_topic}"
        )

    # ─── Dönüş komutu ───
    def on_turn_cmd(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd not in ("left", "right"):
            self.get_logger().warn(f"Geçersiz: '{cmd}' → 'left' veya 'right' gönder")
            return

        if self.state == "WAITING":
            self.pending_turn = cmd
            self.state = "TURNING"
            self.turn_phase = "rotate"
            self.turn_start_time = time.time()
            self.get_logger().info(f"Dönüş başladı: {cmd}")
            self._pub_status("TURNING")
        else:
            self.get_logger().info(
                f"Turn komutu ({cmd}) geldi ama state={self.state}"
            )

    # ─── Detection callback ───
    def on_detection(self, msg: String):
        # Dönüş manevrası sürerken
        if self.state == "TURNING":
            self._execute_turn()
            return

        # Geri gelme sürerken
        if self.state == "REVERSING":
            self._execute_reverse()
            return

        try:
            det = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        found = det.get("found", False)
        cx = det.get("cx", 0)
        w = det.get("w", 1)
        is_intersection = det.get("is_intersection", False)

        base_speed = int(self.get_parameter("base_speed").value)
        kp = float(self.get_parameter("kp").value)
        max_speed = int(self.get_parameter("max_speed").value)
        max_turn = int(self.get_parameter("max_turn").value)
        deadband = int(self.get_parameter("err_deadband_px").value)
        smoothing = float(self.get_parameter("turn_smoothing").value)
        invert = bool(self.get_parameter("invert_steering").value)
        confirm = int(self.get_parameter("intersection_confirm_frames").value)
        cam_offset = int(self.get_parameter("camera_offset_px").value)

        # ── Kesişim algılama ──
        if is_intersection and self.state == "FOLLOWING":
            self.intersection_counter += 1
        elif not is_intersection:
            self.intersection_counter = 0

        if self.intersection_counter >= confirm and self.state == "FOLLOWING":
            # Önce geri gel, sonra dur
            self.state = "REVERSING"
            self.reverse_start_time = time.time()
            self._pub_status("REVERSING")
            self.get_logger().info("KESİŞİM! Geri geliniyor...")
            return

        if self.state == "WAITING":
            self._pub_motor(0, 0)
            return

        # ── Normal şerit takibi ──
        if not found:
            self._pub_motor(0, 0)
            self.integral = 0.0
            return

        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)

        # Hedef: ekran ortası + kamera offset
        target = (w // 2) - cam_offset
        err_px = cx - target
        if abs(err_px) <= deadband:
            err_px = 0

        err = err_px / float(w // 2) if w > 0 else 0.0

        # PID
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now

        self.integral = clamp(self.integral + err * dt, -1.0, 1.0)
        derivative = (err - self.prev_err) / dt
        self.prev_err = err

        turn_raw = int(kp * err + ki * self.integral + kd * derivative)
        if invert:
            turn_raw = -turn_raw
        if max_turn > 0:
            turn_raw = clamp(turn_raw, -max_turn, max_turn)

        smoothing = clamp(smoothing, 0.0, 0.95)
        turn = int((1.0 - smoothing) * turn_raw + smoothing * self.prev_turn)
        self.prev_turn = turn

        left = clamp(base_speed - turn, -max_speed, max_speed)
        right = clamp(base_speed + turn, -max_speed, max_speed)

        self._pub_motor(left, right)

    # ─── Geri gelme (kesişimi ortala) ───
    def _execute_reverse(self):
        rev_speed = int(self.get_parameter("reverse_speed").value)
        rev_dur = float(self.get_parameter("reverse_duration").value)
        elapsed = time.time() - self.reverse_start_time

        if elapsed < rev_dur:
            self._pub_motor(-rev_speed, -rev_speed)
        else:
            self._pub_motor(0, 0)
            self.state = "WAITING"
            self._pub_status("WAITING_AT_INTERSECTION")
            self.get_logger().info(
                "Kesişim ortalandı — komut bekliyor: "
                "ros2 topic pub --once /lane/turn_cmd std_msgs/String \"data: left\"  (veya right)"
            )

    # ─── Dönüş manevrası (ileri-sağ / ileri-sol) ───
    def _execute_turn(self):
        inner = int(self.get_parameter("turn_speed_inner").value)
        outer = int(self.get_parameter("turn_speed_outer").value)
        turn_dur = float(self.get_parameter("turn_duration").value)
        fwd_dur = float(self.get_parameter("forward_after_turn").value)
        base = int(self.get_parameter("base_speed").value)

        elapsed = time.time() - self.turn_start_time

        if self.turn_phase == "rotate":
            if elapsed < turn_dur:
                if self.pending_turn == "left":
                    self._pub_motor(inner, outer)   # sol yavaş, sağ hızlı → sola kıvır
                else:
                    self._pub_motor(outer, inner)   # sol hızlı, sağ yavaş → sağa kıvır
            else:
                self.turn_phase = "forward"
                self.turn_start_time = time.time()

        elif self.turn_phase == "forward":
            if elapsed < fwd_dur:
                self._pub_motor(base, base)
            else:
                self._pub_motor(0, 0)
                self.state = "FOLLOWING"
                self.intersection_counter = 0
                self.pending_turn = None
                self.turn_phase = None
                self.prev_turn = 0
                self.integral = 0.0
                self.prev_err = 0.0
                self._pub_status("FOLLOWING")
                self.get_logger().info("Dönüş tamamlandı → şerit takibine devam")

    # ─── Helpers ───
    def _pub_motor(self, left: int, right: int):
        msg = Int16MultiArray()
        msg.data = [int(left), int(right)]
        self.motor_pub.publish(msg)

    def _pub_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"STATE → {status}")


def main():
    rclpy.init()
    node = LaneControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
