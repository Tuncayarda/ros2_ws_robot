#!/usr/bin/env python3
import time
import smbus2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

ADS1115_ADDR = 0x48
REG_CONVERT = 0x00
REG_CONFIG  = 0x01

# A0 single-ended, FS=±4.096V, 128 SPS
CONFIG_A0 = 0xC383


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class BatteryADS1115Node(Node):

    def __init__(self):
        super().__init__("battery_ads1115_node")

        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("i2c_addr", 72)   # 0x48
        self.declare_parameter("divider_ratio", 5.0)

        self.declare_parameter("ema_alpha", 0.3)
        self.declare_parameter("publish_hz", 2.0)

        bus_id = self.get_parameter("i2c_bus").value
        self.addr = self.get_parameter("i2c_addr").value
        self.bus = smbus2.SMBus(bus_id)

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/battery_status",
            10
        )

        self.filtered_v = None
        period = 1.0 / float(self.get_parameter("publish_hz").value)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info("Battery ADS1115 node READY")

    def read_adc(self):
        self.bus.write_i2c_block_data(
            self.addr,
            REG_CONFIG,
            [(CONFIG_A0 >> 8) & 0xFF, CONFIG_A0 & 0xFF]
        )
        time.sleep(0.008)

        data = self.bus.read_i2c_block_data(self.addr, REG_CONVERT, 2)
        raw = (data[0] << 8) | data[1]
        if raw & 0x8000:
            raw -= 1 << 16
        return raw

    def voltage_to_percent(self, v):
        v = clamp(v, 12.0, 16.5)
        return (v - 12.0) / (16.5 - 12.0)

    def tick(self):
        raw = self.read_adc()

        v_adc = raw * (4.096 / 32768.0)
        v_batt = v_adc * self.get_parameter("divider_ratio").value
        v_batt = clamp(v_batt, 0.0, 16.5)

        alpha = self.get_parameter("ema_alpha").value
        if self.filtered_v is None:
            self.filtered_v = v_batt
        else:
            self.filtered_v = alpha * v_batt + (1 - alpha) * self.filtered_v

        percent = self.voltage_to_percent(self.filtered_v) * 100.0

        msg = Float32MultiArray()
        msg.data = [
            float(percent),       # index 0
            float(self.filtered_v)  # index 1
        ]
        self.pub.publish(msg)

        self.get_logger().debug(
            f"Vbat={self.filtered_v:.2f}V  SOC={percent:.1f}%"
        )


def main():
    rclpy.init()
    node = BatteryADS1115Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()