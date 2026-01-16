#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights


def rosimg_to_bgr(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    step = int(msg.step)
    enc = (msg.encoding or "").lower()

    data = np.frombuffer(msg.data, dtype=np.uint8)
    row = data.reshape((h, step))

    if ("bgra" in enc) or ("bgrx" in enc) or ("xbgr" in enc):
        buf = row[:, : w * 4].reshape((h, w, 4))
        return cv2.cvtColor(buf, cv2.COLOR_BGRA2BGR)

    if ("rgba" in enc) or ("rgbx" in enc) or ("xrgb" in enc):
        buf = row[:, : w * 4].reshape((h, w, 4))
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    if ("bgr8" in enc) or ("bgr888" in enc) or (enc == "bgr"):
        buf = row[:, : w * 3].reshape((h, w, 3))
        return buf.copy()

    if ("rgb8" in enc) or ("rgb888" in enc) or (enc == "rgb"):
        buf = row[:, : w * 3].reshape((h, w, 3))
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    buf = row[:, : w * 3].reshape((h, w, 3))
    return buf.copy()


def apply_rotation(img_bgr: np.ndarray, rotate_deg: int) -> np.ndarray:
    d = int(rotate_deg) % 360
    if d == 0:
        return img_bgr
    if d == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr


class LaneMaskNodeMinimal(Node):
    def __init__(self):
        super().__init__("lane_mask_node")

        default_model = os.path.expanduser(
            "~/ros2_ws_robot/src/lane_inference/model/lane_lraspp_mnv3_best_iou.pth"
        )
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("in_topic",  "/camera/image_raw")
        self.declare_parameter("out_topic", "/lane_mask")

        self.declare_parameter("img_w", 320)
        self.declare_parameter("img_h", 240)
        self.declare_parameter("thresh", 0.50)

        # perf logging
        self.declare_parameter("log_infer_ms", True)
        self.declare_parameter("log_every_n", 30)

        # CPU perf options
        self.declare_parameter("torch_num_threads", 4)
        self.declare_parameter("torch_num_interop_threads", 1)
        self.declare_parameter("use_channels_last", True)
        self.declare_parameter("use_torchscript", False)

        # ✅ rotation (default 180)
        self.declare_parameter("rotate_deg", 180)  # 0/90/180/270

        self.model_path = str(self.get_parameter("model_path").value)
        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.img_w = int(self.get_parameter("img_w").value)
        self.img_h = int(self.get_parameter("img_h").value)
        self.thresh = float(self.get_parameter("thresh").value)
        self.rotate_deg = int(self.get_parameter("rotate_deg").value) % 360
        self.log_infer_ms = bool(self.get_parameter("log_infer_ms").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        torch_num_threads = int(self.get_parameter("torch_num_threads").value)
        torch_num_interop = int(self.get_parameter("torch_num_interop_threads").value)
        self.use_channels_last = bool(self.get_parameter("use_channels_last").value)
        self.use_torchscript = bool(self.get_parameter("use_torchscript").value)

        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
        if torch_num_interop > 0:
            torch.set_num_interop_threads(torch_num_interop)

        if self.rotate_deg not in (0, 90, 180, 270):
            raise ValueError("rotate_deg must be one of 0,90,180,270")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=2,
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=True)
        self.model.eval()

        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.device.type == "cpu" and self.use_torchscript:
            example = torch.zeros((1, 3, self.img_h, self.img_w), dtype=torch.float32)
            if self.use_channels_last:
                example = example.to(memory_format=torch.channels_last)
            self.model = torch.jit.trace(self.model, example, strict=False)
            self.model = torch.jit.optimize_for_inference(self.model)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(Image, self.out_topic, 1)
        self.sub = self.create_subscription(Image, self.in_topic, self.cb, qos)

        self.get_logger().info(
            f"READY | device={self.device} | model={self.model_path} | rotate_deg={self.rotate_deg} | sub={self.in_topic} -> pub={self.out_topic}"
        )

        self._frame_i = 0

    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
        if self.use_channels_last:
            t = t.to(memory_format=torch.channels_last)
        return t.to(self.device)

    def publish_mono8(self, src: Image, mono_u8: np.ndarray):
        out = Image()
        out.header = src.header
        out.height = int(mono_u8.shape[0])
        out.width = int(mono_u8.shape[1])
        out.encoding = "mono8"
        out.step = int(mono_u8.shape[1])
        out.data = mono_u8.tobytes()
        self.pub.publish(out)

    def cb(self, msg: Image):
        try:
            bgr = rosimg_to_bgr(msg)

            # ✅ rotate input first
            bgr = apply_rotation(bgr, self.rotate_deg)

            inp = self.preprocess(bgr)

            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(inp)["out"]
                logits = F.interpolate(
                    logits, size=(self.img_h, self.img_w),
                    mode="bilinear", align_corners=False
                )
                probs = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
                mask_small = (probs >= self.thresh).astype(np.uint8) * 255
            t1 = time.perf_counter()

            # back to rotated image size
            h0, w0 = bgr.shape[:2]
            mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)

            self.publish_mono8(msg, mask_full)

            if self.log_infer_ms:
                self._frame_i += 1
                if self._frame_i % max(1, self.log_every_n) == 0:
                    infer_ms = (t1 - t0) * 1000.0
                    self.get_logger().info(f"lane_mask infer: {infer_ms:.2f} ms")

        except Exception as e:
            self.get_logger().error(f"cb failed: {e}")


def main():
    rclpy.init()
    node = LaneMaskNodeMinimal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()