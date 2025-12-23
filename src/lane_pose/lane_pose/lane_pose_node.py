#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


def circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per <= 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (per * per)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def skeletonize(bin_img: np.ndarray) -> np.ndarray:
    """
    bin_img: 0/255
    returns: skeleton 0/255
    OpenCV-only morphological skeletonization (no ximgproc).
    """
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def line_angle_deg(x1, y1, x2, y2) -> float:
    ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    # normalize to (-90..+90]
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180
    return ang


def dominant_line(lines):
    """
    lines: list of (x1,y1,x2,y2)
    returns longest line (x1,y1,x2,y2) or None
    """
    best = None
    best_len = -1.0
    for (x1, y1, x2, y2) in lines:
        L = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if L > best_len:
            best_len = L
            best = (x1, y1, x2, y2)
    return best


def intersect_lines(p1, p2, p3, p4):
    """
    Infinite line intersection of p1->p2 and p3->p4.
    Returns (cx,cy) float or None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def contour_rect_score(cnt):
    """
    Rectangle-likeness score based on area fill ratio in minAreaRect box.
    """
    area = cv2.contourArea(cnt)
    if area < 1.0:
        return 0.0, None
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    (cx, cy), (rw, rh), ang = rect
    box_area = max(1.0, float(rw * rh))
    fill = float(area / box_area)  # close to 1 => very rectangular (filled)
    return fill, rect


def box_points_from_rect(rect):
    pts = cv2.boxPoints(rect)
    return np.int32(pts)


def rect_center(rect):
    (cx, cy), (rw, rh), ang = rect
    return float(cx), float(cy)


def rect_size(rect):
    (_, _), (rw, rh), _ = rect
    return float(rw), float(rh)


class LanePoseNode(Node):
    """
    /lane_pose (Vector3):
      x = offset_norm (-1..+1)
      y = angle_deg
      z = conf (0..1)

    /lane_intersection (Vector3):
      x = in_intersection (0/1)
      y = mode_hint (-1 ambiguous, 0 vertical-dominant, 1 horizontal-dominant)
      z = progress (0..1)

    /lane_cross (Float32MultiArray):
      data = [in_cross(0/1), cx_norm, cy_norm, ang_h_deg, ang_v_deg, score]
    """

    def __init__(self):
        super().__init__("lane_pose_node")

        # topics
        self.declare_parameter("image_topic", "/camera_bottom/camera_node/image_raw")
        self.declare_parameter("publish_topic", "/lane_pose")

        # rotate in this node
        self.declare_parameter("rotate_deg", 0)

        # intersection
        self.declare_parameter("detect_intersection", True)
        self.declare_parameter("publish_intersection_topic", "/lane_intersection")

        # cross axes (for +)
        self.declare_parameter("detect_cross_axes", True)
        self.declare_parameter("publish_cross_topic", "/lane_cross")

        # cross extraction params (ROI around band bbox)
        self.declare_parameter("cross_band_pad", 18)
        self.declare_parameter("cross_hough_thr", 40)
        self.declare_parameter("cross_min_line_len", 40)
        self.declare_parameter("cross_max_gap", 10)
        self.declare_parameter("cross_need_min_total_lines", 2)

        # ✅ NEW: + rectangle detection + center stripe
        self.declare_parameter("plus_rect_min_area", 700)       # ROI contour area threshold
        self.declare_parameter("plus_rect_fill_thr", 0.55)      # min fill ratio in minAreaRect
        self.declare_parameter("plus_center_tol", 0.12)         # center distance tolerance (fraction of outer min(w,h))
        self.declare_parameter("plus_inner_ratio_min", 0.25)    # inner size ratio vs outer
        self.declare_parameter("plus_inner_ratio_max", 0.80)
        self.declare_parameter("plus_strip_thickness", 14)      # pixels (in ROI coords)
        self.declare_parameter("plus_strip_margin", 10)         # keep stripe away from outer border a bit

        # debug
        self.declare_parameter("debug_view", True)
        self.declare_parameter("debug_window", "lane_pose_debug")
        self.declare_parameter("show_bw", True)

        # grayscale & BW
        self.declare_parameter("blur_ksize", 7)
        self.declare_parameter("use_invert_bw", False)
        self.declare_parameter("clahe_clip", 2.0)
        self.declare_parameter("clahe_tile", 8)

        # LED removal
        self.declare_parameter("led_min_area", 60)
        self.declare_parameter("led_max_area", 12000)
        self.declare_parameter("led_min_circ", 0.45)
        self.declare_parameter("led_dilate", 21)
        self.declare_parameter("led_percentile", 99.3)
        self.declare_parameter("led_min_thr", 200)

        # sampling region
        self.declare_parameter("roi_y0", 0.35)
        self.declare_parameter("roi_y1", 0.95)
        self.declare_parameter("sample_step", 6)
        self.declare_parameter("min_points", 12)

        # row filtering
        self.declare_parameter("row_min_white", 30)
        self.declare_parameter("row_min_span", 60)

        # measurement line
        self.declare_parameter("target_y", 0.70)

        # 2 stripes handling
        self.declare_parameter("prefer_two_walls", True)

        # intersection params
        self.declare_parameter("int_roi_y0", 0.25)
        self.declare_parameter("int_roi_y1", 0.85)
        self.declare_parameter("int_sample_step", 4)
        self.declare_parameter("int_baseline_tail_ratio", 0.25)
        self.declare_parameter("int_width_factor", 1.55)
        self.declare_parameter("int_wide_ratio_thr", 0.55)
        self.declare_parameter("int_on_count", 2)
        self.declare_parameter("int_off_count", 4)
        self.declare_parameter("int_band_half_h", 22)

        # ---- read params
        self.topic = self.get_parameter("image_topic").value
        self.out_topic = self.get_parameter("publish_topic").value

        self.rotate_deg = int(self.get_parameter("rotate_deg").value)

        self.detect_intersection = bool(self.get_parameter("detect_intersection").value)
        self.int_topic = self.get_parameter("publish_intersection_topic").value

        self.detect_cross_axes = bool(self.get_parameter("detect_cross_axes").value)
        self.cross_topic = self.get_parameter("publish_cross_topic").value

        self.cross_band_pad = int(self.get_parameter("cross_band_pad").value)
        self.cross_hough_thr = int(self.get_parameter("cross_hough_thr").value)
        self.cross_min_line_len = int(self.get_parameter("cross_min_line_len").value)
        self.cross_max_gap = int(self.get_parameter("cross_max_gap").value)
        self.cross_need_min_total_lines = int(self.get_parameter("cross_need_min_total_lines").value)

        # ✅ NEW
        self.plus_rect_min_area = int(self.get_parameter("plus_rect_min_area").value)
        self.plus_rect_fill_thr = float(self.get_parameter("plus_rect_fill_thr").value)
        self.plus_center_tol = float(self.get_parameter("plus_center_tol").value)
        self.plus_inner_ratio_min = float(self.get_parameter("plus_inner_ratio_min").value)
        self.plus_inner_ratio_max = float(self.get_parameter("plus_inner_ratio_max").value)
        self.plus_strip_thickness = int(self.get_parameter("plus_strip_thickness").value)
        self.plus_strip_margin = int(self.get_parameter("plus_strip_margin").value)

        self.debug_view = bool(self.get_parameter("debug_view").value)
        self.win = self.get_parameter("debug_window").value
        self.show_bw = bool(self.get_parameter("show_bw").value)

        self.blur_ksize = int(self.get_parameter("blur_ksize").value) | 1
        self.use_invert_bw = bool(self.get_parameter("use_invert_bw").value)
        self.clahe_clip = float(self.get_parameter("clahe_clip").value)
        self.clahe_tile = int(self.get_parameter("clahe_tile").value)

        self.led_min_area = float(self.get_parameter("led_min_area").value)
        self.led_max_area = float(self.get_parameter("led_max_area").value)
        self.led_min_circ = float(self.get_parameter("led_min_circ").value)
        self.led_dilate = int(self.get_parameter("led_dilate").value)
        self.led_percentile = float(self.get_parameter("led_percentile").value)
        self.led_min_thr = int(self.get_parameter("led_min_thr").value)

        self.roi_y0 = float(self.get_parameter("roi_y0").value)
        self.roi_y1 = float(self.get_parameter("roi_y1").value)
        self.sample_step = int(self.get_parameter("sample_step").value)
        self.min_points = int(self.get_parameter("min_points").value)

        self.row_min_white = int(self.get_parameter("row_min_white").value)
        self.row_min_span = int(self.get_parameter("row_min_span").value)

        self.target_y_ratio = float(self.get_parameter("target_y").value)
        self.prefer_two_walls = bool(self.get_parameter("prefer_two_walls").value)

        self.int_roi_y0 = float(self.get_parameter("int_roi_y0").value)
        self.int_roi_y1 = float(self.get_parameter("int_roi_y1").value)
        self.int_sample_step = int(self.get_parameter("int_sample_step").value)
        self.int_baseline_tail_ratio = float(self.get_parameter("int_baseline_tail_ratio").value)
        self.int_width_factor = float(self.get_parameter("int_width_factor").value)
        self.int_wide_ratio_thr = float(self.get_parameter("int_wide_ratio_thr").value)
        self.int_on_count = int(self.get_parameter("int_on_count").value)
        self.int_off_count = int(self.get_parameter("int_off_count").value)
        self.int_band_half_h = int(self.get_parameter("int_band_half_h").value)

        # ---- ros
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Vector3, self.out_topic, 10)
        self.pub_int = self.create_publisher(Vector3, self.int_topic, 10)
        self.pub_cross = self.create_publisher(Float32MultiArray, self.cross_topic, 10)
        self.sub = self.create_subscription(Image, self.topic, self.cb, 10)

        # intersection hysteresis state
        self._int_state = False
        self._int_on_ctr = 0
        self._int_off_ctr = 0
        self._int_progress = 0.0
        self._int_mode_hint = -1

        # last computed cross info for debug overlay
        self._cross_dbg = None

        self.has_display = bool(os.environ.get("DISPLAY"))
        if self.debug_view and self.has_display:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            if self.show_bw:
                cv2.namedWindow(self.win + "_bw", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Sub: {self.topic}")
        self.get_logger().info(f"Pub lane: {self.out_topic}")
        self.get_logger().info(f"Pub int:  {self.int_topic} detect={self.detect_intersection}")
        self.get_logger().info(f"Pub cross:{self.cross_topic} detect={self.detect_cross_axes}")
        self.get_logger().info(f"prefer_two_walls: {self.prefer_two_walls}")
        self.get_logger().info(f"rotate_deg: {self.rotate_deg}")

    # ---------------- core ----------------

    def cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # rotate here
        if self.rotate_deg == 90:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_deg == 180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        elif self.rotate_deg == 270:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = bgr.shape[:2]
        y_target = int(self.target_y_ratio * h)

        # gray + blur
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_tile, self.clahe_tile)
        )
        g = clahe.apply(gray)

        # OTSU bw
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.use_invert_bw:
            bw = 255 - bw

        bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        # ----- LED mask (GRAY-based)
        thr = int(np.clip(np.percentile(g, self.led_percentile), self.led_min_thr, 255))
        led_bin = cv2.inRange(g, thr, 255)
        led_bin = cv2.morphologyEx(
            led_bin, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1
        )

        led_mask = np.zeros((h, w), dtype=np.uint8)
        cnts, _ = cv2.findContours(led_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.led_min_area or area > self.led_max_area:
                continue
            if circularity(c) < self.led_min_circ:
                continue
            cv2.drawContours(led_mask, [c], -1, 255, -1)

        if self.led_dilate > 0:
            led_mask = cv2.dilate(
                led_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.led_dilate, self.led_dilate)),
                iterations=1
            )

        # remove LEDs from bw
        mask = bw.copy()
        mask[led_mask > 0] = 0

        # close to connect surfaces
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
            iterations=2
        )

        # ---- intersection detection (width-profile)
        int_dbg = self._update_intersection(mask)

        # ---- cross axes extraction (ONLY when + exists)
        self._cross_dbg = self._update_cross_axes(mask, int_dbg)

        # ---- SAMPLE rows -> lane centerline (unchanged)
        y0 = int(np.clip(self.roi_y0, 0.0, 1.0) * h)
        y1 = int(np.clip(self.roi_y1, 0.0, 1.0) * h)
        step = max(1, self.sample_step)

        ys, xc = [], []
        xl_dbg, xr_dbg = [], []

        for yy in range(y0, y1, step):
            idx = np.where(mask[yy] > 0)[0]
            if len(idx) < self.row_min_white:
                continue

            gaps = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, gaps + 1]
            ends = np.r_[gaps, len(idx) - 1]

            segments = []
            for s, e in zip(starts, ends):
                l = int(idx[s])
                r = int(idx[e])
                span = r - l
                if span >= self.row_min_span:
                    segments.append((l, r, span))

            if not segments:
                continue

            if self.prefer_two_walls and len(segments) >= 2:
                segments.sort(key=lambda t: t[0])
                left = segments[0]
                right = segments[-1]
                l = left[0]
                r = right[1]
            else:
                segments.sort(key=lambda t: t[2], reverse=True)
                l, r, _ = segments[0]

            x_center = 0.5 * (l + r)

            ys.append(float(yy))
            xc.append(float(x_center))
            xl_dbg.append(float(l))
            xr_dbg.append(float(r))

        if len(xc) < self.min_points:
            self._publish_lane(0.0, 0.0, 0.0)
            self._publish_intersection()
            self._publish_cross_default()
            self._debug(
                bgr, bw, led_mask, mask, None, None, y_target,
                f"NO FIT pts={len(xc)} thr={thr} INT={int(self._int_state)} m={self._int_mode_hint} p={self._int_progress:.2f}",
                int_dbg=int_dbg
            )
            return

        ys_np = np.array(ys, dtype=np.float32)
        xc_np = np.array(xc, dtype=np.float32)

        # fit: x = a*y + b
        a, b = np.polyfit(ys_np, xc_np, 1)

        angle_deg = math.degrees(math.atan(a))

        x_at = float(a * y_target + b)
        offset_px = x_at - (w / 2.0)
        offset_norm = float(offset_px / (w / 2.0))

        conf = float(np.clip(len(xc) / 60.0, 0.0, 1.0))
        self._publish_lane(offset_norm, angle_deg, conf)
        self._publish_intersection()

        # publish cross msg (if found), else defaults
        if self._cross_dbg and self._cross_dbg.get("in_cross", 0) == 1:
            self._publish_cross(self._cross_dbg)
        else:
            self._publish_cross_default()

        text = (
            f"pts={len(xc)} thr={thr} "
            f"off_px={offset_px:.1f} off_n={offset_norm:.3f} "
            f"ang={angle_deg:.2f} conf={conf:.2f}  "
            f"INT={int(self._int_state)} m={self._int_mode_hint} p={self._int_progress:.2f}"
        )

        fit_info = (a, b, x_at)
        samples = (
            ys_np, xc_np,
            np.array(xl_dbg, dtype=np.float32),
            np.array(xr_dbg, dtype=np.float32)
        )
        self._debug(bgr, bw, led_mask, mask, fit_info, samples, y_target, text, int_dbg=int_dbg)

    # ---------------- intersection (width-profile) ----------------

    def _update_intersection(self, mask: np.ndarray):
        if not self.detect_intersection:
            self._int_state = False
            self._int_mode_hint = -1
            self._int_progress = 0.0
            return None

        h, w = mask.shape[:2]
        y0 = int(np.clip(self.int_roi_y0, 0.0, 1.0) * h)
        y1 = int(np.clip(self.int_roi_y1, 0.0, 1.0) * h)
        if y1 <= y0 + 20:
            return None

        step = max(1, self.int_sample_step)

        spans = []
        rows = []
        lefts = []
        rights = []
        whites = []

        for yy in range(y0, y1, step):
            idx = np.where(mask[yy] > 0)[0]
            whites.append(len(idx))
            if len(idx) < self.row_min_white:
                spans.append(0)
                rows.append(yy)
                lefts.append(0)
                rights.append(0)
                continue

            l = int(idx[0])
            r = int(idx[-1])
            spans.append(r - l)
            rows.append(yy)
            lefts.append(l)
            rights.append(r)

        spans_np = np.array(spans, dtype=np.float32)
        rows_np = np.array(rows, dtype=np.int32)
        lefts_np = np.array(lefts, dtype=np.int32)
        rights_np = np.array(rights, dtype=np.int32)

        # baseline from tail rows
        n = len(spans_np)
        tail = max(5, int(self.int_baseline_tail_ratio * n))
        baseline = float(np.median(spans_np[-tail:]))

        peak_i = int(np.argmax(spans_np))
        peak_w = float(spans_np[peak_i])
        peak_y = int(rows_np[peak_i])

        wide_ratio = peak_w / max(1.0, float(w))
        hit = (baseline > 1.0 and peak_w > baseline * self.int_width_factor and wide_ratio > self.int_wide_ratio_thr)

        # estimate intersection band bbox
        band_thr = max(baseline * self.int_width_factor, 0.85 * peak_w)
        band_mask = spans_np >= band_thr
        if np.any(band_mask):
            band_rows = rows_np[band_mask]
            band_y0 = int(band_rows.min())
            band_y1 = int(band_rows.max())
            band_x0 = int(lefts_np[band_mask].min())
            band_x1 = int(rights_np[band_mask].max())
        else:
            band_y0 = peak_y - 10
            band_y1 = peak_y + 10
            band_x0 = int(lefts_np[peak_i])
            band_x1 = int(rights_np[peak_i])

        band_y0 = max(0, band_y0)
        band_y1 = min(h - 1, band_y1)
        band_x0 = max(0, band_x0)
        band_x1 = min(w - 1, band_x1)

        # mode hint around peak_y
        bh = max(8, self.int_band_half_h)
        yy0 = max(0, peak_y - bh)
        yy1 = min(h, peak_y + bh)
        band = mask[yy0:yy1, :]

        b01 = (band > 0).astype(np.uint8)
        row_sum = b01.sum(axis=1).astype(np.float32)
        col_sum = b01.sum(axis=0).astype(np.float32)

        h_strength = float(np.percentile(row_sum, 99) / max(1.0, float(w)))
        v_strength = float(np.percentile(col_sum, 99) / max(1.0, float(b01.shape[0])))

        if max(h_strength, v_strength) < 0.15:
            mode_hint = -1
        else:
            ratio = (v_strength + 1e-6) / (h_strength + 1e-6)
            if 0.80 <= ratio <= 1.25:
                mode_hint = -1
            elif ratio > 1.25:
                mode_hint = 0
            else:
                mode_hint = 1

        # hysteresis update
        if hit:
            self._int_off_ctr = 0
            self._int_on_ctr += 1
            if self._int_on_ctr >= self.int_on_count:
                self._int_state = True
        else:
            self._int_on_ctr = 0
            self._int_off_ctr += 1
            if self._int_off_ctr >= self.int_off_count:
                self._int_state = False

        self._int_mode_hint = mode_hint

        # progress inside band
        if self._int_state and (band_y1 > band_y0 + 5):
            y_now = int(self.target_y_ratio * h)
            self._int_progress = clamp01((y_now - band_y0) / float(band_y1 - band_y0))
        else:
            self._int_progress = 0.0

        return {
            "roi_y0": y0, "roi_y1": y1,
            "baseline": baseline,
            "peak_w": peak_w,
            "peak_y": peak_y,
            "wide_ratio": wide_ratio,
            "hit": hit,
            "band": (band_x0, band_y0, band_x1, band_y1),
            "mode_hint": mode_hint,
            "h_strength": h_strength,
            "v_strength": v_strength,
            "state": self._int_state,
            "progress": self._int_progress,
        }

    # ---------------- + detection helpers ----------------

    def _detect_plus_nested_rects(self, roi_bin_0_255: np.ndarray):
        """
        Finds outer+inner rectangles (minAreaRect) that are nested.
        Returns (outer_rect, inner_rect) or (None, None)
        """
        img = roi_bin_0_255.copy()

        # stabilize contours a bit
        img = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1
        )

        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.plus_rect_min_area:
                continue
            fill, rect = contour_rect_score(c)
            if rect is None:
                continue
            if fill < self.plus_rect_fill_thr:
                continue
            (cx, cy), (rw, rh), ang = rect
            if rw < 10 or rh < 10:
                continue
            cands.append((area, fill, rect))

        if len(cands) < 2:
            return None, None

        # outer = biggest area
        cands.sort(key=lambda t: t[0], reverse=True)
        outer = cands[0][2]
        ocx, ocy = rect_center(outer)
        ow, oh = rect_size(outer)
        omin = max(1.0, min(ow, oh))
        tol = self.plus_center_tol * omin

        # choose best inner: centered + size ratio
        inner_best = None
        best_score = -1.0

        for (_, fill, rect) in cands[1:]:
            icx, icy = rect_center(rect)
            iw, ih = rect_size(rect)

            # center closeness
            d = math.hypot(icx - ocx, icy - ocy)
            if d > tol:
                continue

            # size ratios (use min side)
            rmin = min(iw, ih) / max(1.0, min(ow, oh))
            rmax = max(iw, ih) / max(1.0, max(ow, oh))
            if not (self.plus_inner_ratio_min <= rmin <= self.plus_inner_ratio_max):
                continue
            if not (self.plus_inner_ratio_min <= rmax <= self.plus_inner_ratio_max):
                continue

            score = float(fill) - 0.2 * (d / max(1.0, tol))
            if score > best_score:
                best_score = score
                inner_best = rect

        if inner_best is None:
            return None, None

        return outer, inner_best

    def _build_plus_center_stripe(self, roi_shape, outer_rect, inner_rect):
        """
        Creates a '+' stripe mask centered at rect center, within outer rect.
        Returns stripe_mask (0/255) in ROI coords.
        """
        rh, rw = roi_shape[:2]
        stripe = np.zeros((rh, rw), dtype=np.uint8)

        ocx, ocy = rect_center(outer_rect)
        ow, oh = rect_size(outer_rect)

        cx = int(np.clip(ocx, 0, rw - 1))
        cy = int(np.clip(ocy, 0, rh - 1))

        # stripe extents limited by outer rect size
        half_w = max(10, int(0.5 * min(ow, rw)) - self.plus_strip_margin)
        half_h = max(10, int(0.5 * min(oh, rh)) - self.plus_strip_margin)
        t = max(2, int(self.plus_strip_thickness))

        # vertical bar
        x0 = int(np.clip(cx - t // 2, 0, rw - 1))
        x1 = int(np.clip(cx + t // 2, 0, rw - 1))
        y0 = int(np.clip(cy - half_h, 0, rh - 1))
        y1 = int(np.clip(cy + half_h, 0, rh - 1))
        cv2.rectangle(stripe, (x0, y0), (x1, y1), 255, -1)

        # horizontal bar
        x0 = int(np.clip(cx - half_w, 0, rw - 1))
        x1 = int(np.clip(cx + half_w, 0, rw - 1))
        y0 = int(np.clip(cy - t // 2, 0, rh - 1))
        y1 = int(np.clip(cy + t // 2, 0, rh - 1))
        cv2.rectangle(stripe, (x0, y0), (x1, y1), 255, -1)

        # subtract inner rect region so stripe doesn't get confused by inner frame
        inner_fill = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(inner_fill, [box_points_from_rect(inner_rect)], -1, 255, -1)
        stripe[inner_fill > 0] = 0

        return stripe

    # ---------------- + axes detection (NEW logic) ----------------

    def _update_cross_axes(self, mask: np.ndarray, int_dbg):
        """
        NEW:
          - only when intersection state is active
          - in band ROI: find TWO nested rectangles (outer+inner) => this is "+ exists"
          - remove those rectangles from ROI
          - draw a center stripe (+-shaped) at exact middle
          - skeleton + Hough => axes + center
        """
        if not self.detect_cross_axes:
            return None
        if not self._int_state:
            return None
        if not int_dbg or "band" not in int_dbg:
            return None

        h, w = mask.shape[:2]
        x0, y0, x1, y1 = int_dbg["band"]
        pad = max(0, self.cross_band_pad)

        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w - 1, x1 + pad)
        ry1 = min(h - 1, y1 + pad)

        if rx1 <= rx0 + 10 or ry1 <= ry0 + 10:
            return {"in_cross": 0}

        roi = mask[ry0:ry1, rx0:rx1]
        if roi is None or roi.size == 0:
            return {"in_cross": 0}

        # --- 1) detect + by nested rectangles
        outer_rect, inner_rect = self._detect_plus_nested_rects(roi)
        if outer_rect is None or inner_rect is None:
            # ✅ + yoksa hiçbir şey deneme
            return {"in_cross": 0}

        # --- 2) remove rectangles from roi
        roi_clean = roi.copy()
        rect_mask = np.zeros_like(roi_clean)
        cv2.drawContours(rect_mask, [box_points_from_rect(outer_rect)], -1, 255, -1)
        cv2.drawContours(rect_mask, [box_points_from_rect(inner_rect)], -1, 255, -1)
        roi_clean[rect_mask > 0] = 0

        # --- 3) add center stripe (+ shaped thin band)
        stripe = self._build_plus_center_stripe(roi_clean.shape, outer_rect, inner_rect)

        # final working mask = (roi_clean OR stripe) but keep it sparse
        work = cv2.bitwise_and(stripe, cv2.bitwise_or(roi_clean, stripe))

        # if stripe became empty due to params
        if cv2.countNonZero(work) < 30:
            return {"in_cross": 0}

        # skeleton
        skel = skeletonize(work)

        # Hough
        lines = cv2.HoughLinesP(
            skel, 1, np.pi / 180.0,
            threshold=self.cross_hough_thr,
            minLineLength=self.cross_min_line_len,
            maxLineGap=self.cross_max_gap
        )
        if lines is None:
            return {"in_cross": 0}

        lines = lines[:, 0, :]
        if len(lines) < self.cross_need_min_total_lines:
            return {"in_cross": 0}

        vertical = []
        horizontal = []
        for (lx1, ly1, lx2, ly2) in lines:
            lx1, ly1, lx2, ly2 = int(lx1), int(ly1), int(lx2), int(ly2)
            ang = line_angle_deg(lx1, ly1, lx2, ly2)
            if abs(ang) > 45:
                vertical.append((lx1, ly1, lx2, ly2))
            else:
                horizontal.append((lx1, ly1, lx2, ly2))

        v = dominant_line(vertical) if vertical else None
        hline = dominant_line(horizontal) if horizontal else None

        # fallback: if one axis missing, still accept center from rectangles but score low
        ocx, ocy = rect_center(outer_rect)
        cx = ocx + rx0
        cy = ocy + ry0

        cx = float(np.clip(cx, 0, w - 1))
        cy = float(np.clip(cy, 0, h - 1))

        cx_norm = (cx - (w / 2.0)) / (w / 2.0)
        cy_norm = (cy - (h / 2.0)) / (h / 2.0)

        if not v or not hline:
            # still report cross center from rectangles
            score = min(1.0, (len(vertical) + len(horizontal)) / 30.0)
            return {
                "in_cross": 1,
                "cx": float(cx), "cy": float(cy),
                "cx_norm": float(cx_norm), "cy_norm": float(cy_norm),
                "ang_h": 0.0, "ang_v": 0.0,
                "score": float(0.35 * score),
                "roi": (rx0, ry0, rx1, ry1),
                "v": v, "h": hline,
                "outer_rect": outer_rect,
                "inner_rect": inner_rect,
                "stripe": stripe
            }

        # if we have both, refine intersection by line intersection
        cxcy = intersect_lines(
            (v[0], v[1]), (v[2], v[3]),
            (hline[0], hline[1]), (hline[2], hline[3])
        )
        if cxcy is not None:
            cx_roi, cy_roi = cxcy
            cx2 = cx_roi + rx0
            cy2 = cy_roi + ry0
            if 0 <= cx2 < w and 0 <= cy2 < h:
                cx, cy = float(cx2), float(cy2)
                cx_norm = (cx - (w / 2.0)) / (w / 2.0)
                cy_norm = (cy - (h / 2.0)) / (h / 2.0)

        ang_v = line_angle_deg(*v)
        ang_h = line_angle_deg(*hline)

        score = min(1.0, (len(vertical) + len(horizontal)) / 20.0)

        return {
            "in_cross": 1,
            "cx": float(cx), "cy": float(cy),
            "cx_norm": float(cx_norm), "cy_norm": float(cy_norm),
            "ang_h": float(ang_h), "ang_v": float(ang_v),
            "score": float(score),
            "roi": (rx0, ry0, rx1, ry1),
            "v": v, "h": hline,
            "outer_rect": outer_rect,
            "inner_rect": inner_rect,
            "stripe": stripe
        }

    # ---------------- publishing ----------------

    def _publish_lane(self, offset_norm, angle_deg, conf):
        m = Vector3()
        m.x = float(offset_norm)
        m.y = float(angle_deg)
        m.z = float(conf)
        self.pub.publish(m)

    def _publish_intersection(self):
        m = Vector3()
        m.x = 1.0 if self._int_state else 0.0
        m.y = float(self._int_mode_hint)
        m.z = float(self._int_progress)
        self.pub_int.publish(m)

    def _publish_cross_default(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_cross.publish(msg)

    def _publish_cross(self, d):
        msg = Float32MultiArray()
        msg.data = [
            1.0,
            float(d.get("cx_norm", 0.0)),
            float(d.get("cy_norm", 0.0)),
            float(d.get("ang_h", 0.0)),
            float(d.get("ang_v", 0.0)),
            float(d.get("score", 0.0)),
        ]
        self.pub_cross.publish(msg)

    # ---------------- debug ----------------

    def _debug(self, bgr, bw, led_mask, mask, fit_info, samples, y_target, text, int_dbg=None):
        if not (self.debug_view and self.has_display):
            return

        h, w = bgr.shape[:2]
        vis = bgr.copy()

        # overlay mask
        m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 1.0, m, 0.35, 0)

        # LED outline
        edges = cv2.Canny(led_mask, 50, 150)
        vis[edges > 0] = (0, 0, 255)

        # image center reference
        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)

        # intersection band overlay
        if int_dbg and "band" in int_dbg:
            bx0, by0, bx1, by1 = int_dbg["band"]
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
            cv2.line(vis, (0, int_dbg["peak_y"]), (w - 1, int_dbg["peak_y"]), (0, 255, 0), 1)

        # cross debug overlay
        if self._cross_dbg and self._cross_dbg.get("in_cross", 0) == 1:
            rx0, ry0, rx1, ry1 = self._cross_dbg["roi"]
            cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (0, 200, 0), 2)

            # draw outer+inner rects if present
            if "outer_rect" in self._cross_dbg and "inner_rect" in self._cross_dbg:
                outer = self._cross_dbg["outer_rect"]
                inner = self._cross_dbg["inner_rect"]
                ob = box_points_from_rect(outer) + np.array([rx0, ry0])
                ib = box_points_from_rect(inner) + np.array([rx0, ry0])
                cv2.drawContours(vis, [ob], -1, (0, 255, 0), 2)
                cv2.drawContours(vis, [ib], -1, (0, 255, 0), 2)

            # show stripe pixels (optional)
            if "stripe" in self._cross_dbg and self._cross_dbg["stripe"] is not None:
                stripe = self._cross_dbg["stripe"]
                # paint stripe in cyan
                yy, xx = np.where(stripe > 0)
                yy = yy + ry0
                xx = xx + rx0
                ok = (0 <= yy) & (yy < h) & (0 <= xx) & (xx < w)
                vis[yy[ok], xx[ok]] = (255, 255, 0)

            v = self._cross_dbg.get("v", None)
            hline = self._cross_dbg.get("h", None)
            if v is not None:
                cv2.line(vis, (v[0] + rx0, v[1] + ry0), (v[2] + rx0, v[3] + ry0), (0, 255, 0), 2)
            if hline is not None:
                cv2.line(vis, (hline[0] + rx0, hline[1] + ry0), (hline[2] + rx0, hline[3] + ry0), (0, 255, 0), 2)

            cx = int(self._cross_dbg["cx"])
            cy = int(self._cross_dbg["cy"])
            cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)

            cxn = self._cross_dbg["cx_norm"]
            cyn = self._cross_dbg["cy_norm"]
            cv2.putText(vis, f"PLUS cx_n={cxn:.2f} cy_n={cyn:.2f} sc={self._cross_dbg.get('score',0):.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # sample points
        if samples is not None:
            ys_np, xc_np, xl_np, xr_np = samples
            for y, x, l, r in zip(ys_np.astype(int), xc_np.astype(int),
                                  xl_np.astype(int), xr_np.astype(int)):
                cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)
                cv2.circle(vis, (int(l), int(y)), 2, (255, 0, 255), -1)
                cv2.circle(vis, (int(r), int(y)), 2, (255, 0, 255), -1)

        # fitted line + target marker
        if fit_info is not None:
            a, b, x_at = fit_info
            x_top = int(a * 0 + b)
            x_bot = int(a * (h - 1) + b)
            cv2.line(vis, (x_top, 0), (x_bot, h - 1), (0, 255, 255), 2)
            cv2.line(vis, (int(x_at), 0), (int(x_at), h - 1), (255, 255, 0), 2)
            cv2.line(vis, (0, int(y_target)), (w - 1, int(y_target)), (255, 255, 255), 1)
            cv2.drawMarker(vis, (int(x_at), int(y_target)), (255, 0, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        cv2.putText(vis, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if int_dbg:
            s = (f"int hit={int_dbg['hit']} base={int_dbg['baseline']:.0f} peak={int_dbg['peak_w']:.0f} "
                 f"wr={int_dbg['wide_ratio']:.2f} hs={int_dbg['h_strength']:.2f} vs={int_dbg['v_strength']:.2f}")
            cv2.putText(vis, s, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.win, vis)

        if self.show_bw:
            bw_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            bw_vis[led_mask > 0] = (0, 0, 255)
            cv2.imshow(self.win + "_bw", bw_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = LanePoseNode()
    try:
        rclpy.spin(node)
    finally:
        if bool(os.environ.get("DISPLAY")):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
