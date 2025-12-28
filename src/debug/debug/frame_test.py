#!/usr/bin/env python3
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
    _HAS_CVBRIDGE = True
except Exception:
    CvBridge = None
    _HAS_CVBRIDGE = False

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights


def apply_rotation(img: np.ndarray, rotate_deg: int) -> np.ndarray:
    d = int(rotate_deg) % 360
    if d == 0:
        return img
    if d == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class JpegPipeOneShotNode(Node):
    """
    1) Kameradan 1 frame al
    2) Sampler ile aynı şekilde JPG kaydet (rotate + quality)
    3) Diskten JPG oku
    4) Model ile mask üret
    5) mask.png (ve opsiyonel overlay.jpg) kaydet
    6) Node kapanır
    """
    def __init__(self):
        super().__init__("lane_mask_jpegpipe_oneshot")

        # ---------- params (sampler ile aynı mantık) ----------
        self.declare_parameter("in_topic", "/camera_bottom/camera_node/image_raw")
        self.declare_parameter("out_dir", "/home/robot/lane_debug/oneshot")
        self.declare_parameter("prefix", "oneshot")
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("rotate_deg", 180)  # sampler default

        # ---------- model params ----------
        self.declare_parameter(
            "model_path",
            os.path.expanduser("~/ros2_ws_robot/src/lane_inference/model/lane_lraspp_mbv3_ft_best_iou.pth")
        )
        self.declare_parameter("img_w", 800)
        self.declare_parameter("img_h", 600)
        self.declare_parameter("num_classes", 2)
        self.declare_parameter("lane_class_id", 1)

        # ham model çıktısı için: argmax
        self.declare_parameter("use_argmax", True)

        # eğer softmax isterse (argmax=False) burası devreye girer
        self.declare_parameter("thresh", 0.60)

        # eğitim jpg pipeline: BGR->RGB
        self.declare_parameter("color_mode", "rgb")  # "rgb" or "bgr"

        # debug çıktı
        self.declare_parameter("save_overlay", True)

        # ---------- read ----------
        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.prefix = str(self.get_parameter("prefix").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.rotate_deg = int(self.get_parameter("rotate_deg").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.img_w = int(self.get_parameter("img_w").value)
        self.img_h = int(self.get_parameter("img_h").value)
        self.num_classes = int(self.get_parameter("num_classes").value)
        self.lane_class_id = int(self.get_parameter("lane_class_id").value)
        self.use_argmax = bool(self.get_parameter("use_argmax").value)
        self.thresh = float(self.get_parameter("thresh").value)
        self.color_mode = str(self.get_parameter("color_mode").value).strip().lower()
        self.save_overlay = bool(self.get_parameter("save_overlay").value)

        os.makedirs(self.out_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if self.color_mode not in ("rgb", "bgr"):
            raise ValueError("color_mode must be 'rgb' or 'bgr'")

        # ---------- cv_bridge ----------
        self.bridge = CvBridge() if _HAS_CVBRIDGE else None
        if self.bridge is None:
            raise RuntimeError("cv_bridge not available on this environment.")

        # ---------- torch/model ----------
        self.device = torch.device("cpu")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.get_logger().info(f"Loading model: {self.model_path}")
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=self.num_classes,
        ).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # warmup
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, self.img_h, self.img_w), dtype=torch.float32)
            _ = self.model(dummy)["out"]

        self._done = False
        self.sub = self.create_subscription(Image, self.in_topic, self.cb, 1)

        self.get_logger().info(
            f"READY one-shot.\n"
            f"Sub: {self.in_topic}\n"
            f"Out: {self.out_dir}\n"
            f"rotate_deg={self.rotate_deg} jpeg_quality={self.jpeg_quality}\n"
            f"model img={self.img_w}x{self.img_h} color_mode={self.color_mode} use_argmax={self.use_argmax}"
        )

    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        # resize (safety)
        if img_bgr.shape[1] != self.img_w or img_bgr.shape[0] != self.img_h:
            img = cv2.resize(img_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        else:
            img = img_bgr

        if self.color_mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
        return t.to(self.device)

    def infer_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        inp = self.preprocess(img_bgr)
        with torch.inference_mode():
            logits = self.model(inp)["out"]  # 1xCxhxw
            if logits.shape[-2] != self.img_h or logits.shape[-1] != self.img_w:
                logits = F.interpolate(logits, size=(self.img_h, self.img_w),
                                       mode="bilinear", align_corners=False)

            if self.use_argmax:
                pred = torch.argmax(logits, dim=1)[0].to(torch.uint8).cpu().numpy()
                mask = (pred == self.lane_class_id).astype(np.uint8) * 255
            else:
                probs = torch.softmax(logits, dim=1)[0, self.lane_class_id].cpu().numpy()
                mask = (probs >= self.thresh).astype(np.uint8) * 255
        return mask

    def cb(self, msg: Image):
        if self._done:
            return
        self._done = True

        try:
            self.get_logger().info(f"Incoming encoding: {msg.encoding} ({msg.width}x{msg.height}) step={msg.step}")

            # ✅ sampler ile aynı: bgr8 isteniyor, sonra rotate
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img_bgr = apply_rotation(img_bgr, self.rotate_deg)

            ts = time.strftime("%Y%m%d_%H%M%S")
            base = f"{ts}_{self.prefix}"

            jpg_path = os.path.join(self.out_dir, f"{base}.jpg")
            mask_path = os.path.join(self.out_dir, f"{base}_mask.png")
            overlay_path = os.path.join(self.out_dir, f"{base}_overlay.jpg")

            # 1) JPG yaz
            ok = cv2.imwrite(jpg_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise RuntimeError(f"Failed to write jpg: {jpg_path}")

            # 2) Diskten aynı JPG oku
            jpg_bgr = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
            if jpg_bgr is None:
                raise RuntimeError(f"Failed to read back jpg: {jpg_path}")

            # 3) Model inference -> mask
            mask = self.infer_mask(jpg_bgr)

            # 4) mask kaydet
            ok = cv2.imwrite(mask_path, mask)
            if not ok:
                raise RuntimeError(f"Failed to write mask: {mask_path}")

            # 5) overlay opsiyonel
            if self.save_overlay:
                img_vis = cv2.resize(jpg_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
                colored = np.zeros_like(img_vis)
                colored[mask > 0] = (0, 255, 0)  # BGR green
                overlay = cv2.addWeighted(img_vis, 1.0, colored, 0.45, 0.0)
                cv2.imwrite(overlay_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            self.get_logger().info(f"SAVED:\n  jpg : {jpg_path}\n  mask: {mask_path}")
            if self.save_overlay:
                self.get_logger().info(f"  ovl : {overlay_path}")

        except Exception as e:
            self.get_logger().error(f"One-shot failed: {e}")

        # node'u kapat
        self.get_logger().info("Done. Shutting down.")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = JpegPipeOneShotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()