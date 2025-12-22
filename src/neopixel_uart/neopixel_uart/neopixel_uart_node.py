#!/usr/bin/env python3
import time
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8, UInt16, UInt8MultiArray

import serial


SOF1 = 0xAA
SOF2 = 0x55
VER  = 0x01

T_SET_CONFIG = 0x01


def clamp_u8(x: int) -> int:
    return max(0, min(255, int(x)))


def crc8(data: bytes) -> int:
    # CRC-8 poly 0x07, init 0x00
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


@dataclass
class LedConfig:
    mode: int = 0
    speed_ms: int = 60
    brightness: int = 180
    A: tuple = (255, 0, 0)
    B: tuple = (0, 0, 255)


class NeoPixelUartNode(Node):
    def __init__(self):
        super().__init__("neopixel_uart_node")

        # ----- Params -----
        self.declare_parameter("port", "/dev/ttyAMA0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("send_hz", 20)          # kaç Hz config yollasın (debounce için)
        self.declare_parameter("log_frames", False)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.send_hz = max(1, int(self.get_parameter("send_hz").value))
        self.log_frames = bool(self.get_parameter("log_frames").value)

        # ----- State -----
        self.cfg = LedConfig()
        self.cfg_lock = threading.Lock()

        self.dirty = False               # config değişti mi?
        self.last_sent = 0.0

        # ----- Serial -----
        self.ser = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0)
        self.get_logger().info(f"Opened UART {self.port} @ {self.baud}")

        # ----- Topics -----
        # 1) mode: UInt8
        self.sub_mode = self.create_subscription(UInt8, "/led/mode", self.cb_mode, 10)

        # 2) speed_ms: UInt16
        self.sub_speed = self.create_subscription(UInt16, "/led/speed_ms", self.cb_speed, 10)

        # 3) brightness: UInt8
        self.sub_bri = self.create_subscription(UInt8, "/led/brightness", self.cb_brightness, 10)

        # 4) colors: UInt8MultiArray, length=6 -> [rA,gA,bA,rB,gB,bB]
        self.sub_colors = self.create_subscription(UInt8MultiArray, "/led/colors", self.cb_colors, 10)

        # timer: düzenli olarak en son config’i gönder
        period = 1.0 / float(self.send_hz)
        self.timer = self.create_timer(period, self.tick)

        # ilk açılışta bir kere gönderelim
        self.mark_dirty()
        self.get_logger().info("Listening topics: /led/mode /led/speed_ms /led/brightness /led/colors")

    # ---------- callbacks ----------
    def mark_dirty(self):
        self.dirty = True

    def cb_mode(self, msg: UInt8):
        with self.cfg_lock:
            self.cfg.mode = int(msg.data)
        self.mark_dirty()

    def cb_speed(self, msg: UInt16):
        with self.cfg_lock:
            self.cfg.speed_ms = int(msg.data)
        self.mark_dirty()

    def cb_brightness(self, msg: UInt8):
        with self.cfg_lock:
            self.cfg.brightness = int(msg.data)
        self.mark_dirty()

    def cb_colors(self, msg: UInt8MultiArray):
        data = list(msg.data)
        if len(data) != 6:
            self.get_logger().warn(f"/led/colors expected len=6 got len={len(data)}")
            return
        rA, gA, bA, rB, gB, bB = map(int, data)
        with self.cfg_lock:
            self.cfg.A = (rA, gA, bA)
            self.cfg.B = (rB, gB, bB)
        self.mark_dirty()

    # ---------- frame ----------
    def build_set_config_frame(self, cfg: LedConfig) -> bytes:
        mode = clamp_u8(cfg.mode)
        speed = max(0, min(65535, int(cfg.speed_ms)))
        bri = clamp_u8(cfg.brightness)

        rA, gA, bA = [clamp_u8(x) for x in cfg.A]
        rB, gB, bB = [clamp_u8(x) for x in cfg.B]

        payload = bytes([
            mode,
            speed & 0xFF, (speed >> 8) & 0xFF,
            bri,
            rA, gA, bA,
            rB, gB, bB
        ])

        ln = len(payload)
        hdr = bytes([SOF1, SOF2, VER, T_SET_CONFIG, ln])
        body = hdr + payload

        c = crc8(body[2:])   # VER..TYPE..LEN..PAYLOAD
        frame = body + bytes([c])
        return frame

    # ---------- send loop ----------
    def tick(self):
        now = time.monotonic()
        if not self.dirty:
            return

        # debounce: aynı timer tick içinde spamlemeyelim
        if (now - self.last_sent) < (1.0 / self.send_hz):
            return

        with self.cfg_lock:
            cfg_copy = LedConfig(
                mode=self.cfg.mode,
                speed_ms=self.cfg.speed_ms,
                brightness=self.cfg.brightness,
                A=self.cfg.A,
                B=self.cfg.B
            )

        frame = self.build_set_config_frame(cfg_copy)

        try:
            self.ser.write(frame)
            self.last_sent = now
            self.dirty = False
            if self.log_frames:
                self.get_logger().info(f"TX {len(frame)} bytes: {frame.hex()}")
        except Exception as e:
            self.get_logger().error(f"UART write failed: {e}")

    def destroy_node(self):
        try:
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = NeoPixelUartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()