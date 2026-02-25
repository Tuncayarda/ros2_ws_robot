#!/usr/bin/env python3
"""
lane_detector_node
──────────────────
Kameradan gelen görüntüyü işler, şerit konumunu ve kesişim bilgisini publish eder.
Motor sürmez — sadece algılama yapar.

Publish:
  /lane/detection   (std_msgs/String)  JSON: {"found", "cx", "w", "white_ratio", "width_ratio", "is_intersection"}
  /lane/debug_image (sensor_msgs/Image) debug görüntü

Subscribe:
  /camera_bottom/camera_node/image_raw  (sensor_msgs/Image)
"""
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__("lane_detector_node")

        # ── Topics ──
        self.declare_parameter("image_topic", "/camera_bottom/camera_node/image_raw")
        self.declare_parameter("detection_topic", "/lane/detection")
        self.declare_parameter("debug_image_topic", "/lane/debug_image")
        self.declare_parameter("publish_debug_image", True)

        # ── ROI ──
        self.declare_parameter("roi_height_ratio", 0.40)
        self.declare_parameter("rotate_180", True)

        # ── Beyaz şerit algılama ──
        self.declare_parameter("blur_ksize", 5)
        self.declare_parameter("clahe_clip", 4.0)
        self.declare_parameter("adaptive_block", 31)
        self.declare_parameter("adaptive_c", -8)
        self.declare_parameter("sat_max", 255)
        self.declare_parameter("val_min", 10)

        # ── Kontur filtreleme ──
        self.declare_parameter("min_contour_area", 200)
        self.declare_parameter("min_aspect_ratio", 0.8)
        self.declare_parameter("morph_kernel", 5)

        # ── Kesişim eşikleri ──
        self.declare_parameter("intersection_white_ratio", 0.30)
        self.declare_parameter("intersection_width_ratio", 0.70)

        # ── IO ──
        self.bridge = CvBridge()

        img_topic = self.get_parameter("image_topic").value
        det_topic = self.get_parameter("detection_topic").value
        dbg_topic = self.get_parameter("debug_image_topic").value

        self.det_pub = self.create_publisher(String, det_topic, 10)
        self.dbg_pub = self.create_publisher(Image, dbg_topic, 10)
        self.sub = self.create_subscription(Image, img_topic, self.on_image, 10)

        self.get_logger().info(
            f"lane_detector READY | sub={img_topic} → det={det_topic} dbg={dbg_topic}"
        )

    # ────────────────────────────────────────────
    def on_image(self, msg: Image):
        roi_h_ratio = float(self.get_parameter("roi_height_ratio").value)
        blur_ksize = int(self.get_parameter("blur_ksize").value)
        clahe_clip = float(self.get_parameter("clahe_clip").value)
        adaptive_block = int(self.get_parameter("adaptive_block").value)
        adaptive_c = int(self.get_parameter("adaptive_c").value)
        sat_max = int(self.get_parameter("sat_max").value)
        val_min = int(self.get_parameter("val_min").value)
        min_contour_area = int(self.get_parameter("min_contour_area").value)
        min_aspect_ratio = float(self.get_parameter("min_aspect_ratio").value)
        k = int(self.get_parameter("morph_kernel").value)
        pub_dbg = bool(self.get_parameter("publish_debug_image").value)
        rotate = bool(self.get_parameter("rotate_180").value)
        inter_wr = float(self.get_parameter("intersection_white_ratio").value)
        inter_wid = float(self.get_parameter("intersection_width_ratio").value)

        # ── Frame al ──
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]
        roi_h = int(h * roi_h_ratio)
        y0 = h - roi_h
        roi = frame[y0:h, 0:w]

        # ── Mask oluştur ──
        mask = self._build_mask(roi, blur_ksize, clahe_clip, adaptive_block,
                                adaptive_c, sat_max, val_min, k,
                                min_contour_area, min_aspect_ratio)

        # ── Kesişim oranları ──
        roi_total_px = mask.shape[0] * mask.shape[1]
        white_px = cv2.countNonZero(mask)
        white_ratio = white_px / float(roi_total_px) if roi_total_px > 0 else 0.0

        # Yatay yayilim: beyaz piksellerin x eksenindeki dagilimi
        col_sum = np.sum(mask > 0, axis=0)
        active_cols = np.sum(col_sum > (mask.shape[0] * 0.05))
        width_ratio = active_cols / float(w) if w > 0 else 0.0

        # Kesisim: birden fazla buyuk kontur varsa veya beyaz alan genis yayilmissa
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        num_big = len(big_contours)

        # Birden fazla buyuk kontur + genis yayilim = kesisim
        is_intersection = (
            (white_ratio > inter_wr and width_ratio > inter_wid)
            or (num_big >= 2 and width_ratio > 0.50)
        )

        # ── En büyük kontur → şerit merkezi ──
        found = False
        cx = w // 2

        if big_contours:
            largest = max(big_contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                found = True

        # ── Detection publish ──
        det = {
            "found": bool(found),
            "cx": int(cx),
            "w": int(w),
            "white_ratio": round(float(white_ratio), 4),
            "width_ratio": round(float(width_ratio), 4),
            "is_intersection": bool(is_intersection),
        }
        det_msg = String()
        det_msg.data = json.dumps(det)
        self.det_pub.publish(det_msg)

        # ── Debug image ──
        if pub_dbg:
            self._publish_debug(frame, roi, mask, cx if found else None,
                                w // 2, white_ratio, width_ratio,
                                is_intersection, found)

    # ────────────────────────────────────────────
    def _build_mask(self, roi, blur_ksize, clahe_clip, adaptive_block,
                    adaptive_c, sat_max, val_min, k,
                    min_contour_area, min_aspect_ratio):
        """Resmin geri kalanina gore parlak (beyaz) olan pikselleri bul."""
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blur_ksize = max(3, blur_ksize)
        rh, rw = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # Ortalama + std tabanli dinamik esik
        mean_val = np.mean(gray)
        std_val = max(np.std(gray), 1.0)
        factor = abs(adaptive_c) / 10.0
        factor = max(0.3, min(factor, 2.0))
        thresh = int(mean_val + std_val * factor)
        thresh = max(thresh, 80)

        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # Morfoloji
        k = max(3, k)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Kenar bolgeleri sil (sol/sag %5 serit olamaz, genelde lens distortion)
        edge_px = max(int(rw * 0.03), 2)
        mask[:, :edge_px] = 0
        mask[:, rw - edge_px:] = 0

        # Kontur filtrele: alan + solidity (ici dolu olma orani)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.3:
                continue
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
        return filtered

    # ────────────────────────────────────────────
    def _publish_debug(self, frame, roi, mask, cx, target,
                       white_ratio, width_ratio, is_inter, found):
        h, w = frame.shape[:2]
        roi_h = roi.shape[0]
        y0 = h - roi_h

        dbg = frame.copy()
        cv2.line(dbg, (0, y0), (w, y0), (255, 255, 0), 1)
        cv2.line(dbg, (target, y0), (target, h), (0, 255, 0), 2)

        if cx is not None:
            cv2.circle(dbg, (cx, y0 + roi_h // 2), 6, (0, 0, 255), -1)

        # Durum yazısı
        if is_inter:
            label = f"INTERSECTION w={white_ratio:.0%} wd={width_ratio:.0%}"
            color = (0, 0, 255)
        elif found:
            label = f"LANE cx={cx} w={white_ratio:.0%}"
            color = (0, 255, 0)
        else:
            label = "NO LANE"
            color = (0, 165, 255)

        cv2.putText(dbg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mask inset sag ust
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inset_h = min(roi_h, 150)
        if mask.shape[0] > 0:
            inset_w = max(1, int(inset_h * (mask.shape[1] / mask.shape[0])))
            inset = cv2.resize(mask_bgr, (inset_w, inset_h))
            dbg[0:inset_h, w - inset_w:w] = inset

        try:
            img_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            self.dbg_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"debug img: {e}")


def main():
    rclpy.init()
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
