#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
    _HAS_CVBRIDGE = True
except Exception:
    _HAS_CVBRIDGE = False
    CvBridge = None


@dataclass
class FrameStore:
    img_bgr: Optional[np.ndarray] = None
    stamp_sec: float = 0.0


class CameraSamplerNode(Node):
    def __init__(self):
        super().__init__("camera_sampler_node")

        # -------- params --------
        self.declare_parameter("topic", "/camera_bottom/camera_node/image_raw")
        self.declare_parameter("out_dir", "/home/robot/datasets/bottom_samples")
        self.declare_parameter("interval_ms", 150)
        self.declare_parameter("default_count", 800)     # default set target
        self.declare_parameter("prefix", "bottom")
        self.declare_parameter("show", True)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("rotate_deg", 180)        # 0, 90, 180, 270
        self.declare_parameter("write_preview", True)    # overlay yerine ham da kaydetmek istersen

        self.topic = self.get_parameter("topic").value
        self.out_dir = self.get_parameter("out_dir").value
        self.interval_ms = int(self.get_parameter("interval_ms").value)
        self.default_count = int(self.get_parameter("default_count").value)
        self.prefix = self.get_parameter("prefix").value
        self.show = bool(self.get_parameter("show").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.rotate_deg = int(self.get_parameter("rotate_deg").value)
        self.write_preview = bool(self.get_parameter("write_preview").value)

        os.makedirs(self.out_dir, exist_ok=True)

        self.bridge = CvBridge() if _HAS_CVBRIDGE else None
        self.frame = FrameStore()

        # ---- capture state ----
        self.armed = False
        self.set_name = None
        self.set_count = 0
        self.total_target = self.default_count

        # Runtime target choices (quick set)
        self.target_presets = [200, 500, 800, 1200, 2000]
        self.target_idx = 2  # default 800
        self.total_target = self.target_presets[self.target_idx]

        # Category map (extend freely)
        self.categories: Dict[str, str] = {
            "1": "straight_lane",
            "2": "side_lane",
            "3": "intersection",
            "n": "negative",

            # ekstra kategoriler
            "4": "curve_lane",
            "5": "t_junction",
            "6": "cross_plus",
            "7": "glare_strong",
            "8": "blur_motion",
            "9": "weird_case",
        }

        # session stamp for unique filenames
        self.session_id = time.strftime("%Y%m%d_%H%M%S")

        self.sub = self.create_subscription(Image, self.topic, self.cb_image, 10)
        self.timer = self.create_timer(self.interval_ms / 1000.0, self.tick)

        self.print_help()

        if self.show:
            cv2.namedWindow("camera_sampler", cv2.WINDOW_NORMAL)

    def print_help(self):
        self.get_logger().info(
            "CameraSampler READY.\n"
            "CATEGORY KEYS:\n"
            "  1: straight_lane\n"
            "  2: side_lane\n"
            "  3: intersection\n"
            "  n: negative\n"
            "  4: curve_lane\n"
            "  5: t_junction\n"
            "  6: cross_plus\n"
            "  7: glare_strong\n"
            "  8: blur_motion\n"
            "  9: weird_case\n"
            "\n"
            "TARGET (how many images in a set):\n"
            "  [-] / [+] : decrease / increase target preset\n"
            f"  presets={self.target_presets}\n"
            "\n"
            "CONTROL:\n"
            "  s: stop/abort current set\n"
            "  SPACE: save ONE frame (single shot)\n"
            "  q or ESC: quit\n"
        )

    def apply_rotation(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        d = self.rotate_deg % 360
        if d == 0:
            return img
        if d == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if d == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if d == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def start_set(self, name: str):
        self.armed = True
        self.set_name = name
        self.set_count = 0
        self.total_target = self.target_presets[self.target_idx]
        self.get_logger().info(f"START set='{self.set_name}' target={self.total_target}")

    def stop_set(self):
        if self.armed:
            self.get_logger().warn(f"STOP set='{self.set_name}' at {self.set_count}/{self.total_target}")
        self.armed = False
        self.set_name = None
        self.set_count = 0

    def decode_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        # Prefer cv_bridge
        if self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                pass

        # Robust fallback
        h, w = msg.height, msg.width
        enc = (msg.encoding or "").lower()
        step = msg.step
        data = np.frombuffer(msg.data, dtype=np.uint8)

        row = data.reshape((h, step))

        if "bgra" in enc or "bgrx" in enc or "xbgr" in enc:
            buf = row[:, : w * 4].reshape((h, w, 4))
            return cv2.cvtColor(buf, cv2.COLOR_BGRA2BGR)

        if "rgba" in enc or "rgbx" in enc or "xrgb" in enc:
            buf = row[:, : w * 4].reshape((h, w, 4))
            return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        if "rgb8" in enc or enc == "rgb":
            buf = row[:, : w * 3].reshape((h, w, 3))
            return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

        # assume bgr8
        buf = row[:, : w * 3].reshape((h, w, 3))
        return buf.copy()

    def cb_image(self, msg: Image):
        img_bgr = self.decode_to_bgr(msg)
        if img_bgr is None:
            return

        img_bgr = self.apply_rotation(img_bgr)
        self.frame.img_bgr = img_bgr
        self.frame.stamp_sec = time.time()

    def save_one(self, img_bgr: np.ndarray, tag: str, idx: int):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.prefix}_{tag}_{self.session_id}_{ts}_{idx:06d}.jpg"
        path = os.path.join(self.out_dir, fname)
        ok = cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            self.get_logger().info(f"saved: {path}")
        else:
            self.get_logger().error(f"FAILED save: {path}")

    def handle_keys(self, key: int):
        # Quit
        if key in (27, ord('q')):
            self.get_logger().info("Quit requested.")
            if self.show:
                cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        # Stop current set
        if key == ord('s'):
            self.stop_set()
            return

        # Single shot
        if key == 32:  # SPACE
            if self.frame.img_bgr is not None:
                self.save_one(self.frame.img_bgr, "single", int(time.time() * 1000) % 1000000)
            return

        # Target preset - / +
        if key in (ord('-'), ord('_')):
            self.target_idx = max(0, self.target_idx - 1)
            self.get_logger().info(f"TARGET preset -> {self.target_presets[self.target_idx]}")
            return
        if key in (ord('+'), ord('=')):
            self.target_idx = min(len(self.target_presets) - 1, self.target_idx + 1)
            self.get_logger().info(f"TARGET preset -> {self.target_presets[self.target_idx]}")
            return

        # Category start
        ch = chr(key) if 0 <= key < 256 else ""
        if ch in self.categories:
            self.start_set(self.categories[ch])
            return

    def tick(self):
        if self.frame.img_bgr is None:
            return

        img = self.frame.img_bgr

        # GUI overlay
        if self.show:
            overlay = img.copy()
            status = "IDLE"
            if self.armed:
                status = f"REC [{self.set_name}] {self.set_count}/{self.total_target}"
            tgt = self.target_presets[self.target_idx]

            cv2.putText(overlay, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if self.armed else (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(overlay, f"TARGET preset: {tgt}  (use - / +)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(overlay, "Keys: 1 2 3 n 4..9 | SPACE=single | s=stop | q=quit",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("camera_sampler", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_keys(key)

        # ---- recording ----
        if not self.armed:
            return

        if self.set_count >= self.total_target:
            self.get_logger().info(f"DONE set='{self.set_name}' {self.total_target}/{self.total_target}")
            self.stop_set()
            return

        self.set_count += 1
        self.save_one(img, self.set_name, self.set_count)


def main():
    rclpy.init()
    node = CameraSamplerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()