#!/usr/bin/env python3
import time
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8, UInt16, Int16, UInt8MultiArray

import serial


# ================= Protocol =================
SOF1 = 0xAA
SOF2 = 0x55
VER  = 0x01

# ROS -> Pico (TX)
T_SET_CONFIG = 0x01

# Pico -> ROS (RX)
T_SENSOR_TELEM = 0x20

# Parser states
S_FIND_SOF1 = 0
S_FIND_SOF2 = 1
S_READ_VER  = 2
S_READ_TYPE = 3
S_READ_LEN  = 4
S_READ_PAY  = 5
S_READ_CRC  = 6


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


def u16_le(b0: int, b1: int) -> int:
    return (b1 << 8) | b0


def i16_le(b0: int, b1: int) -> int:
    v = (b1 << 8) | b0
    return v - 0x10000 if v & 0x8000 else v


@dataclass
class LedConfig:
    mode: int = 0
    speed_ms: int = 60
    brightness: int = 180
    A: tuple = (255, 255, 255)
    B: tuple = (255, 255, 255)


class PicoComNode(Node):
    """
    - Publishes Pico sensor telemetry (topics FIXED, do not change)
    - Sends LED config frames to Pico (AA55 v1 + CRC8)
    """
    def __init__(self):
        super().__init__("pico_com")

        # ------------- Params -------------
        self.declare_parameter("port", "/dev/ttyAMA0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("send_hz", 20)
        self.declare_parameter("rx_hz", 200)
        self.declare_parameter("frame_timeout_ms", 80)
        self.declare_parameter("log_frames", False)
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("auto_reopen", True)
        self.declare_parameter("reopen_sec", 1.0)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.send_hz = max(1, int(self.get_parameter("send_hz").value))
        self.rx_hz = max(10, int(self.get_parameter("rx_hz").value))
        self.frame_timeout_ms = int(self.get_parameter("frame_timeout_ms").value)
        self.log_frames = bool(self.get_parameter("log_frames").value)
        self.publish_raw = bool(self.get_parameter("publish_raw").value)
        self.auto_reopen = bool(self.get_parameter("auto_reopen").value)
        self.reopen_sec = float(self.get_parameter("reopen_sec").value)

        # ------------- Serial -------------
        self.ser_lock = threading.Lock()
        self.ser = None
        self.last_open_try = 0.0
        self._ensure_serial_open(force=True)

        # ------------- LED TX state -------------
        self.cfg = LedConfig()
        self.cfg_lock = threading.Lock()
        self.dirty = True
        self.last_sent = 0.0

        # ------------- ROS Subscribers (LED control) -------------
        self.sub_mode = self.create_subscription(UInt8, "/led/mode", self.cb_mode, 10)
        self.sub_speed = self.create_subscription(UInt16, "/led/speed_ms", self.cb_speed, 10)
        self.sub_bri = self.create_subscription(UInt8, "/led/brightness", self.cb_brightness, 10)
        self.sub_colors = self.create_subscription(UInt8MultiArray, "/led/colors", self.cb_colors, 10)

        # ------------- ROS Publishers (Telemetry) -------------
        # (TOPICS MUST STAY SAME)
        self.pub_flags = self.create_publisher(UInt8, "/pico/telem/flags", 10)
        self.pub_eco2  = self.create_publisher(UInt16, "/pico/ccs811/eco2_ppm", 10)
        self.pub_tvoc  = self.create_publisher(UInt16, "/pico/ccs811/tvoc_ppb", 10)
        self.pub_temp  = self.create_publisher(Int16,  "/pico/dht22/temp_c_x100", 10)
        self.pub_hum   = self.create_publisher(UInt16, "/pico/dht22/hum_pct_x100", 10)
        self.pub_rms   = self.create_publisher(UInt16, "/pico/mic/rms", 10)
        self.pub_peak  = self.create_publisher(UInt16, "/pico/mic/peak", 10)

        self.pub_raw = None
        if self.publish_raw:
            self.pub_raw = self.create_publisher(UInt8MultiArray, "/pico/telem/raw", 10)

        # ------------- Serial RX parser state -------------
        self.state = S_FIND_SOF1
        self.buf = bytearray()   # VER..TYPE..LEN..PAYLOAD (for CRC)
        self.need = 0
        self.last_byte_ms = int(time.monotonic() * 1000)

        # ------------- last values (optional: stability) -------------
        self.last_flags = 0
        self.last_eco2 = 0
        self.last_tvoc = 0
        self.last_temp = 0
        self.last_hum = 0
        self.last_rms = 0
        self.last_peak = 0

        # ------------- Timers -------------
        self.tx_timer = self.create_timer(1.0 / float(self.send_hz), self.tx_tick)
        self.rx_timer = self.create_timer(1.0 / float(self.rx_hz), self.rx_tick)

        self.get_logger().info(
            "Listening: /led/mode /led/speed_ms /led/brightness /led/colors | "
            "Publishing: /pico/* telemetry (fixed topics)"
        )

    # ===================== Serial helpers =====================
    def _ensure_serial_open(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and (now - self.last_open_try) < self.reopen_sec:
            return self.ser is not None

        self.last_open_try = now

        with self.ser_lock:
            if self.ser is not None:
                return True
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0)
                self.get_logger().info(f"Opened UART {self.port} @ {self.baud}")
                return True
            except Exception as e:
                self.ser = None
                self.get_logger().warn(f"UART open failed ({self.port}): {e}")
                return False

    def _serial_write(self, data: bytes) -> bool:
        if self.ser is None:
            if not self.auto_reopen:
                return False
            if not self._ensure_serial_open():
                return False

        with self.ser_lock:
            try:
                self.ser.write(data)
                return True
            except Exception as e:
                self.get_logger().error(f"UART write failed: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return False

    def _serial_read_available(self) -> bytes:
        if self.ser is None:
            if not self.auto_reopen:
                return b""
            if not self._ensure_serial_open():
                return b""

        with self.ser_lock:
            try:
                n = self.ser.in_waiting
                if n <= 0:
                    return b""
                return self.ser.read(n)
            except Exception as e:
                self.get_logger().error(f"UART read failed: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return b""

    # ===================== ROS callbacks (LED config) =====================
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

    # ===================== Frame builder (TX) =====================
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
        frame = bytearray([SOF1, SOF2, VER, T_SET_CONFIG, ln])
        frame.extend(payload)
        c = crc8(frame[2:])  # VER..TYPE..LEN..PAYLOAD
        frame.append(c)
        return bytes(frame)

    def tx_tick(self):
        if not self.dirty:
            return

        now = time.monotonic()
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
        ok = self._serial_write(frame)
        if ok:
            self.last_sent = now
            self.dirty = False
            if self.log_frames:
                self.get_logger().info(f"TX {len(frame)} bytes: {frame.hex()}")

    # ===================== Serial RX parser =====================
    def reset_parser(self):
        self.state = S_FIND_SOF1
        self.buf = bytearray()
        self.need = 0

    def parser_feed_byte(self, b: int):
        self.last_byte_ms = int(time.monotonic() * 1000)

        if self.state == S_FIND_SOF1:
            if b == SOF1:
                self.state = S_FIND_SOF2
            return

        if self.state == S_FIND_SOF2:
            if b == SOF2:
                self.state = S_READ_VER
            else:
                self.state = S_FIND_SOF1
            return

        if self.state == S_READ_VER:
            if b != VER:
                self.reset_parser()
                return
            self.buf = bytearray([b])  # VER
            self.state = S_READ_TYPE
            return

        if self.state == S_READ_TYPE:
            self.buf.append(b)  # TYPE
            self.state = S_READ_LEN
            return

        if self.state == S_READ_LEN:
            self.buf.append(b)  # LEN
            self.need = b
            self.state = S_READ_PAY if self.need > 0 else S_READ_CRC
            return

        if self.state == S_READ_PAY:
            self.buf.append(b)
            self.need -= 1
            if self.need <= 0:
                self.state = S_READ_CRC
            return

        if self.state == S_READ_CRC:
            got = b
            calc = crc8(self.buf)
            if calc == got:
                ftype = self.buf[1]
                ln = self.buf[2]
                payload = self.buf[3:3 + ln] if ln > 0 else b""
                self.handle_frame(ftype, payload)
            elif self.log_frames:
                self.get_logger().warn("RX CRC mismatch (dropped frame)")
            self.reset_parser()
            return

    def handle_frame(self, ftype: int, payload: bytes):
        if self.log_frames:
            self.get_logger().info(f"RX type=0x{ftype:02X} len={len(payload)} payload={payload.hex()}")

        if ftype != T_SENSOR_TELEM:
            return

        # payload: flags(1) eco2(2) tvoc(2) temp(2) hum(2) rms(2) peak(2) => 13
        if len(payload) != 13:
            self.get_logger().warn(f"T_SENSOR_TELEM payload len expected 13 got {len(payload)}")
            return

        flags = payload[0]
        eco2  = u16_le(payload[1], payload[2])
        tvoc  = u16_le(payload[3], payload[4])
        temp  = i16_le(payload[5], payload[6])
        hum   = u16_le(payload[7], payload[8])
        rms   = u16_le(payload[9], payload[10])
        peak  = u16_le(payload[11], payload[12])

        # cache (optional, useful for debug)
        self.last_flags = flags
        self.last_eco2 = eco2
        self.last_tvoc = tvoc
        self.last_temp = temp
        self.last_hum = hum
        self.last_rms = rms
        self.last_peak = peak

        # Publish (topics fixed)
        self.pub_flags.publish(UInt8(data=flags))
        self.pub_eco2.publish(UInt16(data=eco2))
        self.pub_tvoc.publish(UInt16(data=tvoc))
        self.pub_temp.publish(Int16(data=temp))
        self.pub_hum.publish(UInt16(data=hum))
        self.pub_rms.publish(UInt16(data=rms))
        self.pub_peak.publish(UInt16(data=peak))

        if self.pub_raw is not None:
            self.pub_raw.publish(UInt8MultiArray(data=list(payload)))

    def rx_tick(self):
        # mid-frame timeout
        now_ms = int(time.monotonic() * 1000)
        if self.state != S_FIND_SOF1 and (now_ms - self.last_byte_ms) > self.frame_timeout_ms:
            self.reset_parser()

        data = self._serial_read_available()
        if not data:
            return

        for bb in data:
            self.parser_feed_byte(bb)

    def destroy_node(self):
        with self.ser_lock:
            try:
                if self.ser is not None:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
        super().destroy_node()


def main():
    rclpy.init()
    node = PicoComNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()