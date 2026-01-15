#!/usr/bin/env python3
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


def fnum(x, nd=3, default="—"):
    if x is None:
        return default
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return default


def fint(x, default="—"):
    if x is None:
        return default
    try:
        return str(int(x))
    except Exception:
        return default


class LaneInfoViewer(Node):
    def __init__(self):
        super().__init__("lane_info_viewer")
        self.declare_parameter("topic", "/camera_bottom/lane_info")
        self.declare_parameter("print_hz", 10.0)   # max print rate
        self.declare_parameter("show_raw", False)  # print full json too

        topic = str(self.get_parameter("topic").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(String, topic, self.cb, qos)
        self.last_print_t = 0.0
        self.get_logger().info(f"Listening: {topic}")

        # header
        print(
            "time     sel vis int | dx_sel   ang_sel  conf_sel | "
            "dx1      ang1     conf1    | dx2      ang2     conf2"
        )

    def pick(self, d, key):
        v = d.get(key, None)
        if isinstance(v, dict):
            return v
        return None

    def cb(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception:
            return

        now = time.time()
        min_dt = 1.0 / max(1.0, float(self.get_parameter("print_hz").value))
        if now - self.last_print_t < min_dt:
            return
        self.last_print_t = now

        lane_sel = d.get("lane_select", {}) if isinstance(d.get("lane_select", {}), dict) else {}
        sel_id = lane_sel.get("selected_lane_id", None)
        sel_vis = lane_sel.get("selected_visible", None)
        inter = d.get("intersection", None)

        lane1 = self.pick(d, "lane1")
        lane2 = self.pick(d, "lane2")
        selected = self.pick(d, "selected")

        # Use center_offset_x_norm as main dx; fallback bottom_intersect_x_norm if missing
        def dx_of(x):
            if not isinstance(x, dict):
                return None
            return x.get("center_offset_x_norm", x.get("bottom_intersect_x_norm", None))

        def ang_of(x):
            if not isinstance(x, dict):
                return None
            return x.get("angle_to_red_signed_deg", None)

        def conf_of(x):
            if not isinstance(x, dict):
                return None
            return x.get("confidence", None)

        tstr = time.strftime("%H:%M:%S")

        line = (
            f"{tstr}  "
            f"{fint(sel_id):>3} {str(sel_vis):>3} {str(inter):>3} | "
            f"{fnum(dx_of(selected),3):>7} {fnum(ang_of(selected),1):>7} {fnum(conf_of(selected),2):>8} | "
            f"{fnum(dx_of(lane1),3):>7} {fnum(ang_of(lane1),1):>7} {fnum(conf_of(lane1),2):>8} | "
            f"{fnum(dx_of(lane2),3):>7} {fnum(ang_of(lane2),1):>7} {fnum(conf_of(lane2),2):>8}"
        )
        print(line)

        if bool(self.get_parameter("show_raw").value):
            print(json.dumps(d, indent=2, ensure_ascii=False))


def main():
    rclpy.init()
    node = LaneInfoViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()