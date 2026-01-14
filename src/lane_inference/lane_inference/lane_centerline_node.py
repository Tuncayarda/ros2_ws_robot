#!/usr/bin/env python3
import time
import json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

W_TARGET = 320
H_TARGET = 240

OUTLINE_K = 5
EDGE_THICKNESS = 2
BORDER_ERASE = 3

HOUGH_THRESH = 22
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 10
ANGLE_MERGE_DEG = 12

MIN_PARALLEL_SEP = 18.0
MAX_PARALLEL_SEP = 120.0
MIN_GROUP_LEN = 40.0
MIN_T_OVERLAP = 0.20

DRAW_SINGLE_EDGE_CANDIDATE = True
SINGLE_EDGE_MIN_TOTAL_LEN = 90.0
SINGLE_EDGE_MIN_SPAN_T = 70.0

MAX_CENTERLINES = 2
CENTER_THICKNESS = 2

LANE_EDGE_COLOR = (0, 140, 255)     # BGR orange
LANE_EDGE_THICKNESS = 2
DRAW_LINE_TYPE = cv2.LINE_8

COLOR_LANE1 = (0, 255, 0)   # green
COLOR_LANE2 = (255, 0, 0)   # blue


# =========================================================
# LINE UTILS
# =========================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def extend_to_borders(x1, y1, x2, y2, w, h, scale=10000):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    xa = int(round(x1 - dx * scale))
    ya = int(round(y1 - dy * scale))
    xb = int(round(x1 + dx * scale))
    yb = int(round(y1 + dy * scale))
    ok, p1, p2 = cv2.clipLine((0, 0, w, h), (xa, ya), (xb, yb))
    if not ok:
        return None
    return (p1[0], p1[1], p2[0], p2[1])

def _angle_deg(x1, y1, x2, y2):
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    ang = (ang + 180.0) % 180.0
    return float(ang)

def _ang_dist(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)

def angle_to_vertical_signed_deg(x1, y1, x2, y2) -> float:
    # signed relative to vertical axis
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    ang = np.degrees(np.arctan2(dx, dy))  # vertical reference
    if ang > 90.0:
        ang -= 180.0
    if ang < -90.0:
        ang += 180.0
    return float(ang)

def line_x_at_y(x1, y1, x2, y2, y_query):
    y1f, y2f = float(y1), float(y2)
    if abs(y2f - y1f) < 1e-6:
        return None
    t = (float(y_query) - y1f) / (y2f - y1f)
    return float(x1) + t * float(x2 - x1)


# =========================================================
# CLUSTERING / STRUCTS
# =========================================================
def _cluster_by_angle(lines, tol_deg):
    clusters = []

    def _robust_mean_angle(items):
        s = 0.0
        c = 0.0
        for it in items:
            ang = np.radians(it[4])
            w = float(it[5])
            c += np.cos(2.0 * ang) * w
            s += np.sin(2.0 * ang) * w
        mean = 0.5 * np.degrees(np.arctan2(s, c))
        mean = (mean + 180.0) % 180.0
        return float(mean)

    for L in lines:
        placed = False
        for cl in clusters:
            if _ang_dist(L[4], cl["mean"]) <= tol_deg:
                cl["items"].append(L)
                cl["mean"] = _robust_mean_angle(cl["items"])
                placed = True
                break
        if not placed:
            clusters.append({"mean": float(L[4]), "items": [L]})

    for cl in clusters:
        cl["mean"] = _robust_mean_angle(cl["items"])
        cl["score"] = sum(it[5] for it in cl["items"])

    clusters.sort(key=lambda z: z["score"], reverse=True)
    return clusters

def _t_range_of_points(pts_xy, ux, uy):
    t = pts_xy[:, 0] * ux + pts_xy[:, 1] * uy
    return float(t.min()), float(t.max())

def _line_from_u_c(ux, uy, nx, ny, c_val, tmin, tmax, w, h):
    xA = ux * tmin + nx * c_val
    yA = uy * tmin + ny * c_val
    xB = ux * tmax + nx * c_val
    yB = uy * tmax + ny * c_val

    x1, y1 = int(round(xA)), int(round(yA))
    x2, y2 = int(round(xB)), int(round(yB))

    ok, p1, p2 = cv2.clipLine((0, 0, w, h), (x1, y1), (x2, y2))
    if not ok:
        return None
    return (p1[0], p1[1], p2[0], p2[1])

def _centerline_and_edges_from_cluster(
    cluster, w, h,
    min_parallel_sep, max_parallel_sep, min_group_len, min_t_overlap,
    draw_single_edge, single_edge_min_total_len, single_edge_min_span_t
):
    theta = np.radians(cluster["mean"])
    ux, uy = float(np.cos(theta)), float(np.sin(theta))
    nx, ny = -uy, ux

    items = cluster["items"]
    if len(items) < 2:
        return None

    segs = []
    pts_all = []
    for (x1, y1, x2, y2, ang, L) in items:
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        c = nx * mx + ny * my
        segs.append((x1, y1, x2, y2, float(L), float(c)))
        pts_all.append((x1, y1))
        pts_all.append((x2, y2))

    c_vals = np.array([s[5] for s in segs], dtype=np.float32)
    if c_vals.size < 2:
        return None

    total_len = sum(s[4] for s in segs)

    pts_all = np.array(pts_all, dtype=np.float32)
    tmin_all, tmax_all = _t_range_of_points(pts_all, ux, uy)
    span_t_all = tmax_all - tmin_all

    c_spread = float(c_vals.max() - c_vals.min())

    two_side = False
    center_line = None
    edge_lines = []

    if c_spread >= min_parallel_sep:
        c_sorted = np.sort(c_vals)
        mu1 = float(c_sorted[int(0.25 * (len(c_sorted) - 1))])
        mu2 = float(c_sorted[int(0.75 * (len(c_sorted) - 1))])

        lab = None
        for _ in range(10):
            d1 = np.abs(c_vals - mu1)
            d2 = np.abs(c_vals - mu2)
            lab = (d2 < d1).astype(np.int32)

            if np.all(lab == 0) or np.all(lab == 1):
                break

            mu1_new = float(c_vals[lab == 0].mean())
            mu2_new = float(c_vals[lab == 1].mean())
            if abs(mu1_new - mu1) < 1e-3 and abs(mu2_new - mu2) < 1e-3:
                mu1, mu2 = mu1_new, mu2_new
                break
            mu1, mu2 = mu1_new, mu2_new

        if lab is not None:
            g1 = [segs[i] for i in range(len(segs)) if lab[i] == 0]
            g2 = [segs[i] for i in range(len(segs)) if lab[i] == 1]

            len1 = sum(s[4] for s in g1)
            len2 = sum(s[4] for s in g2)

            sep = abs(mu2 - mu1)

            if (len1 >= min_group_len and len2 >= min_group_len and
                min_parallel_sep <= sep <= max_parallel_sep):

                c1 = sum(s[5] * s[4] for s in g1) / (len1 + 1e-9)
                c2 = sum(s[5] * s[4] for s in g2) / (len2 + 1e-9)
                c_mid = 0.5 * (c1 + c2)

                def _group_pts(group):
                    pts = []
                    for (x1, y1, x2, y2, L, c) in group:
                        pts.append((x1, y1))
                        pts.append((x2, y2))
                    return np.array(pts, dtype=np.float32)

                pts1 = _group_pts(g1)
                pts2 = _group_pts(g2)
                t1min, t1max = _t_range_of_points(pts1, ux, uy)
                t2min, t2max = _t_range_of_points(pts2, ux, uy)

                ov = max(0.0, min(t1max, t2max) - max(t1min, t2min))
                span = max(1e-6, max(t1max, t2max) - min(t1min, t2min))
                if (ov / span) >= min_t_overlap:
                    tmin = min(t1min, t2min)
                    tmax = max(t1max, t2max)

                    e1 = _line_from_u_c(ux, uy, nx, ny, c1, tmin, tmax, w, h)
                    e2 = _line_from_u_c(ux, uy, nx, ny, c2, tmin, tmax, w, h)
                    cc = _line_from_u_c(ux, uy, nx, ny, c_mid, tmin, tmax, w, h)

                    if e1 is not None and e2 is not None and cc is not None:
                        two_side = True
                        edge_lines = [e1, e2]
                        center_line = cc

    if not two_side and draw_single_edge:
        if total_len >= single_edge_min_total_len and span_t_all >= single_edge_min_span_t:
            c_hat = sum(s[5] * s[4] for s in segs) / (total_len + 1e-9)
            e = _line_from_u_c(ux, uy, nx, ny, c_hat, tmin_all, tmax_all, w, h)
            if e is not None:
                edge_lines = [e]
                center_line = None
            else:
                return None
        else:
            return None

    if not edge_lines and center_line is None:
        return None

    return {"angle": float(cluster["mean"]), "center": center_line, "edges": edge_lines}

def edge_to_structures(edge255: np.ndarray, params: dict):
    h, w = edge255.shape
    linesP = cv2.HoughLinesP(
        edge255, 1, np.pi / 180.0, params["hough_thresh"],
        minLineLength=params["hough_min_len"], maxLineGap=params["hough_max_gap"]
    )
    if linesP is None:
        return []

    lines = []
    for l in linesP[:, 0]:
        x1, y1, x2, y2 = map(int, l)
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L < 10.0:
            continue
        ang = _angle_deg(x1, y1, x2, y2)
        lines.append((x1, y1, x2, y2, ang, L))

    if not lines:
        return []

    clusters = _cluster_by_angle(lines, params["angle_merge_deg"])

    structs = []
    for c in clusters:
        st = _centerline_and_edges_from_cluster(
            c, w, h,
            params["min_parallel_sep"], params["max_parallel_sep"],
            params["min_group_len"], params["min_t_overlap"],
            params["draw_single_edge"],
            params["single_edge_min_total_len"], params["single_edge_min_span_t"],
        )
        if st is None:
            continue

        ok = True
        for ex in structs:
            if _ang_dist(st["angle"], ex["angle"]) < 18.0:
                ok = False
                break
        if ok:
            structs.append(st)
        if len(structs) >= params["max_centerlines"]:
            break

    return structs


# =========================
# PREPROCESS
# =========================
class PreprocCache:
    def __init__(self, outline_k: int, edge_thickness: int):
        self.k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_k, outline_k))
        self.k_thick = None
        if edge_thickness > 1:
            self.k_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thickness, edge_thickness))

    def filled_to_outline_255(self, mask255: np.ndarray) -> np.ndarray:
        grad = cv2.morphologyEx(mask255, cv2.MORPH_GRADIENT, self.k_grad)
        out = (grad > 0).astype(np.uint8) * 255
        if self.k_thick is not None:
            out = cv2.dilate(out, self.k_thick)
        return out

    @staticmethod
    def erase_border_inplace(bin255: np.ndarray, px: int):
        if px <= 0:
            return
        h, w = bin255.shape
        px = int(px)
        px = max(0, min(px, min(h, w) // 2))
        bin255[:px, :] = 0
        bin255[-px:, :] = 0
        bin255[:, :px] = 0
        bin255[:, -px:] = 0


# =========================================================
# METRICS / FEATURE
# =========================================================
def compute_confidence(s: dict) -> float:
    edges_count = len(s.get("edges", []))
    has_pair = (edges_count == 2)
    center_present = (s.get("center") is not None)

    ang = float(s.get("angle", 0.0))
    ang_err = _ang_dist(ang, 90.0)
    angle_score = max(0.0, 1.0 - (ang_err / 45.0))

    score = 0.0
    score += 0.55 if has_pair else 0.15
    score += 0.25 if center_present else 0.0
    score += 0.20 * angle_score
    return float(max(0.0, min(1.0, score)))

def _feat_from_line(line, w, h, cx, y_bottom, y_mid):
    x1, y1, x2, y2 = line
    ext = extend_to_borders(x1, y1, x2, y2, w, h)
    if ext is not None:
        x1, y1, x2, y2 = ext

    ang_deg = _angle_deg(x1, y1, x2, y2)
    ang_to_vert_signed = angle_to_vertical_signed_deg(x1, y1, x2, y2)

    xb = line_x_at_y(x1, y1, x2, y2, y_bottom)
    xm = line_x_at_y(x1, y1, x2, y2, y_mid)
    if xb is None or xm is None:
        return None

    xb = float(clamp(xb, 0.0, float(w - 1)))
    xm = float(clamp(xm, 0.0, float(w - 1)))

    bottom_dx_px = xb - float(cx)
    mid_dx_px = xm - float(cx)

    bottom_dx_norm = float(bottom_dx_px / (w * 0.5 + 1e-9))
    mid_dx_norm = float(mid_dx_px / (w * 0.5 + 1e-9))

    mid_x_seg = 0.5 * (x1 + x2)
    center_offset_px = float(mid_x_seg - cx)
    center_offset_norm = float(center_offset_px / (w * 0.5 + 1e-9))

    return {
        "angle_deg": float(ang_deg),
        "angle_to_red_signed_deg": float(ang_to_vert_signed),
        "bottom_intersect_x_px": xb,
        "bottom_intersect_dx_px": bottom_dx_px,
        "bottom_intersect_x_norm": bottom_dx_norm,
        "mid_intersect_x_px": xm,
        "mid_intersect_dx_px": mid_dx_px,
        "mid_intersect_x_norm": mid_dx_norm,
        "center_offset_x_px": center_offset_px,
        "center_offset_x_norm": center_offset_norm,
        "line": (int(x1), int(y1), int(x2), int(y2)),
        "angle_to_vertical_abs_err": float(_ang_dist(ang_deg, 90.0)),
    }

def lane_feat_from_struct_center(s, w, h):
    cx = w // 2
    y_bottom = h - 1
    y_mid = h // 2

    center = s.get("center", None)
    if center is None:
        return None

    base = _feat_from_line(center, w, h, cx, y_bottom, y_mid)
    if base is None:
        return None

    edges_count = int(len(s.get("edges", [])))
    has_pair = bool(edges_count == 2)
    conf = float(compute_confidence(s))

    base.update({
        "confidence": conf,
        "has_pair": has_pair,
        "edges_count": edges_count,
        "center": base["line"],
    })
    return base

def edge_feat_from_struct_single(s, w, h, lane_half_width_px: float):
    """
    centerline yokken tek turuncu edge'i kullan:
    - edge'in bottom dx'i
    - şeridi bulmak için önerilen arama yönü
    - varsayımsal lane_half_width_px ile tahmini merkez dx
    """
    cx = w // 2
    y_bottom = h - 1
    y_mid = h // 2

    edges = s.get("edges", [])
    if len(edges) != 1:
        return None

    base = _feat_from_line(edges[0], w, h, cx, y_bottom, y_mid)
    if base is None:
        return None

    edge_bottom_dx_px = float(base["bottom_intersect_dx_px"])
    edge_bottom_dx_norm = float(base["bottom_intersect_x_norm"])

    # Edge soldaysa (neg dx) muhtemelen sol sınır -> merkez daha sağda -> sağa "ara"
    # Edge sağdaysa (pos dx) muhtemelen sağ sınır -> merkez daha solda -> sola "ara"
    if edge_bottom_dx_px < 0.0:
        search_dir = "right"
        est_center_dx_px = edge_bottom_dx_px + float(lane_half_width_px)
    else:
        search_dir = "left"
        est_center_dx_px = edge_bottom_dx_px - float(lane_half_width_px)

    est_center_dx_norm = float(est_center_dx_px / (w * 0.5 + 1e-9))

    return {
        "confidence": float(compute_confidence(s)),
        "edge_bottom_dx_px": edge_bottom_dx_px,
        "edge_bottom_dx_norm": edge_bottom_dx_norm,
        "search_dir": search_dir,
        "estimated_center_dx_px": float(est_center_dx_px),
        "estimated_center_dx_norm": float(est_center_dx_norm),
        "edge_line": base["line"],
        "angle_deg": float(base["angle_deg"]),
        "angle_to_red_signed_deg": float(base["angle_to_red_signed_deg"]),
    }


# =========================================================
# MAIN PROCESS
# =========================================================
def process_one(mask255: np.ndarray, params: dict, cache: PreprocCache, vis: np.ndarray):
    edge = cache.filled_to_outline_255(mask255)
    cache.erase_border_inplace(edge, params["border_erase"])
    structs = edge_to_structures(edge, params)

    h, w = edge.shape
    vis.fill(0)

    # red center line
    cx = w // 2
    cv2.line(vis, (cx, 0), (cx, h - 1), (0, 0, 255), 1, DRAW_LINE_TYPE)

    # draw edges first
    for D in structs:
        for (x1, y1, x2, y2) in D["edges"]:
            ext = extend_to_borders(x1, y1, x2, y2, w, h)
            if ext is None:
                continue
            ex1, ey1, ex2, ey2 = ext
            cv2.line(vis, (ex1, ey1), (ex2, ey2), LANE_EDGE_COLOR, LANE_EDGE_THICKNESS, DRAW_LINE_TYPE)

    return edge, vis, structs


# =========================================================
# TURN / LOCK STATE
# =========================================================
def _norm(v: float) -> float:
    return float(max(-1.0, min(1.0, v)))

def _parse_turn_request(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("left", "right", "straight", "none"):
        return s
    return "none"


class LaneCenterlineNode320x240(Node):
    """
    Sub:  /camera_bottom/lane_mask        (mono8 0/255)  -> 320x240
    Pub:  /camera_bottom/center_lines     (bgr8)         -> 320x240 visualization
    Pub:  /camera_bottom/lane_info        (String JSON)  -> every frame
    Sub:  /camera_bottom/turn_request     (String)       -> "left|right|straight|none"
    Sub:  /camera_bottom/turn_done        (String)       -> "true/1" -> unlock
    """
    def __init__(self):
        super().__init__("lane_centerline_node")

        self.declare_parameter("in_topic", "/camera_bottom/lane_mask")
        self.declare_parameter("out_topic", "/camera_bottom/center_lines")
        self.declare_parameter("info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("turn_request_topic", "/camera_bottom/turn_request")
        self.declare_parameter("turn_done_topic", "/camera_bottom/turn_done")
        self.declare_parameter("skip_n", 0)

        # single-edge heuristic
        self.declare_parameter("lane_half_width_px", 60.0)

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)
        self.turn_request_topic = str(self.get_parameter("turn_request_topic").value)
        self.turn_done_topic = str(self.get_parameter("turn_done_topic").value)
        self.skip_n = int(self.get_parameter("skip_n").value)
        self.lane_half_width_px = float(self.get_parameter("lane_half_width_px").value)

        self.params = {
            "border_erase": BORDER_ERASE,
            "hough_thresh": HOUGH_THRESH,
            "hough_min_len": HOUGH_MIN_LEN,
            "hough_max_gap": HOUGH_MAX_GAP,
            "angle_merge_deg": float(ANGLE_MERGE_DEG),

            "min_parallel_sep": float(MIN_PARALLEL_SEP),
            "max_parallel_sep": float(MAX_PARALLEL_SEP),
            "min_group_len": float(MIN_GROUP_LEN),
            "min_t_overlap": float(MIN_T_OVERLAP),

            "draw_single_edge": bool(DRAW_SINGLE_EDGE_CANDIDATE),
            "single_edge_min_total_len": float(SINGLE_EDGE_MIN_TOTAL_LEN),
            "single_edge_min_span_t": float(SINGLE_EDGE_MIN_SPAN_T),

            "max_centerlines": int(MAX_CENTERLINES),
            "center_thickness": int(CENTER_THICKNESS),
        }

        self.cache = PreprocCache(OUTLINE_K, EDGE_THICKNESS)
        self.bridge = CvBridge()
        self._vis = np.zeros((H_TARGET, W_TARGET, 3), np.uint8)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(Image, self.in_topic, self.cb, qos)
        self.pub_img = self.create_publisher(Image, self.out_topic, qos)
        self.pub_info = self.create_publisher(String, self.info_topic, qos)

        self.sub_turn = self.create_subscription(String, self.turn_request_topic, self.cb_turn_request, qos)
        self.sub_done = self.create_subscription(String, self.turn_done_topic, self.cb_turn_done, qos)

        # state
        self._i = 0
        self._t_fps = time.time()
        self._n_fps = 0

        self.turn_request = "none"   # left|right|straight|none
        self.turn_state = "IDLE"     # IDLE|LOCKED
        self.locked_lane_id = None   # 1 or 2
        self.lock_expire_t = 0.0     # safety timeout (optional)

        self.get_logger().info("READY lane_centerline_node (intersection lock + single-edge hints)")
        self.get_logger().info(f"Sub  : {self.in_topic}")
        self.get_logger().info(f"Pub  : {self.out_topic}")
        self.get_logger().info(f"Info : {self.info_topic}")
        self.get_logger().info(f"Turn : {self.turn_request_topic} / {self.turn_done_topic}")

    def cb_turn_request(self, msg: String):
        req = _parse_turn_request(msg.data)
        self.turn_request = req
        # request gelince bir sonraki intersection frame'inde lock atacağız
        # straight: lane1 (merkeze yakın) lock
        self.get_logger().info(f"turn_request = {self.turn_request}")

    def cb_turn_done(self, msg: String):
        s = (msg.data or "").strip().lower()
        done = (s in ("1", "true", "yes", "done", "ok"))
        if done:
            self.turn_state = "IDLE"
            self.locked_lane_id = None
            self.turn_request = "none"
            self.get_logger().info("turn_done -> unlock")

    def _choose_lane_with_turn(self, lane1, lane2):
        """
        lane1/lane2: packed feat dicts (centerline)
        Return (selected_lane_id, selected_feat)
        """
        if lane1 is None and lane2 is None:
            return (None, None)
        if lane2 is None:
            return (1, lane1)
        if lane1 is None:
            return (2, lane2)

        # both exist: intersection candidate
        # decide by request
        if self.turn_request == "left":
            # choose the lane whose bottom dx is more negative (more left)
            if lane1["bottom_intersect_dx_px"] <= lane2["bottom_intersect_dx_px"]:
                return (1, lane1)
            else:
                return (2, lane2)
        if self.turn_request == "right":
            # choose more positive
            if lane1["bottom_intersect_dx_px"] >= lane2["bottom_intersect_dx_px"]:
                return (1, lane1)
            else:
                return (2, lane2)
        # straight or none -> choose closest to center
        if abs(lane1["bottom_intersect_dx_px"]) <= abs(lane2["bottom_intersect_dx_px"]):
            return (1, lane1)
        return (2, lane2)

    def cb(self, msg: Image):
        self._i += 1
        if self.skip_n > 0 and (self._i % (self.skip_n + 1)) != 0:
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        if mask.shape[0] != H_TARGET or mask.shape[1] != W_TARGET:
            self.get_logger().error(f"Mask must be {W_TARGET}x{H_TARGET}, got {mask.shape[1]}x{mask.shape[0]}")
            return

        _, vis, structs = process_one(mask, self.params, self.cache, self._vis)

        # ---- centerline feats
        center_dets = []
        # ---- single-edge dets (centerline olmayanlar)
        single_edges = []

        for s in structs:
            f = lane_feat_from_struct_center(s, W_TARGET, H_TARGET)
            if f is not None:
                center_dets.append(f)
            else:
                e = edge_feat_from_struct_single(s, W_TARGET, H_TARGET, self.lane_half_width_px)
                if e is not None:
                    single_edges.append(e)

        # ---- sort centerline dets by vertical closeness
        center_sorted = sorted(center_dets, key=lambda d: d["angle_to_vertical_abs_err"])
        lane1 = center_sorted[0] if len(center_sorted) >= 1 else None
        lane2 = center_sorted[1] if len(center_sorted) >= 2 else None

        intersection = bool(lane1 is not None and lane2 is not None)

        # ---- lock logic:
        # If LOCKED -> keep locked lane id if available
        # If IDLE and intersection and request != none -> lock chosen lane
        selected_lane_id = None
        selected_lane = None

        if self.turn_state == "LOCKED":
            if self.locked_lane_id == 1 and lane1 is not None:
                selected_lane_id, selected_lane = (1, lane1)
            elif self.locked_lane_id == 2 and lane2 is not None:
                selected_lane_id, selected_lane = (2, lane2)
            else:
                # locked lane not visible -> fall back (do not auto-switch unless you want)
                selected_lane_id, selected_lane = self._choose_lane_with_turn(lane1, lane2)

        else:
            # IDLE
            if intersection and self.turn_request in ("left", "right", "straight"):
                chosen_id, chosen = self._choose_lane_with_turn(lane1, lane2)
                self.turn_state = "LOCKED"
                self.locked_lane_id = chosen_id
                selected_lane_id, selected_lane = (chosen_id, chosen)
            else:
                selected_lane_id, selected_lane = self._choose_lane_with_turn(lane1, lane2)

        # ---- draw centerlines (green/blue) as before
        def draw_center(feat, color):
            if feat is None:
                return
            x1, y1, x2, y2 = feat["center"]
            ext = extend_to_borders(x1, y1, x2, y2, W_TARGET, H_TARGET)
            if ext is None:
                return
            ex1, ey1, ex2, ey2 = ext
            cv2.line(vis, (ex1, ey1), (ex2, ey2), color, self.params["center_thickness"], DRAW_LINE_TYPE)

        draw_center(lane1, COLOR_LANE1)
        draw_center(lane2, COLOR_LANE2)

        # ---- publish vis
        out_img = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_img.header = msg.header
        self.pub_img.publish(out_img)

        # ---- build info
        cx = W_TARGET // 2
        info = {
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "img": {"w": W_TARGET, "h": H_TARGET, "center_x": cx, "bottom_y": H_TARGET - 1, "mid_y": H_TARGET // 2},

            "intersection": bool(intersection),
            "lane_count_centerline": int(len(center_sorted)),
            "lane_count_single_edge": int(len(single_edges)),

            "turn": {
                "request": self.turn_request,
                "state": self.turn_state,
                "locked_lane_id": int(self.locked_lane_id) if self.locked_lane_id is not None else None,
                "selected_lane_id": int(selected_lane_id) if selected_lane_id is not None else None,
            },

            "lane1": None,
            "lane2": None,
            "selected": None,

            "lanes_centerline": [],
            "lanes_single_edge": [],
        }

        def pack_center(feat, lane_id: int):
            if feat is None:
                return None
            return {
                "id": lane_id,
                "confidence": float(feat["confidence"]),
                "has_pair": bool(feat["has_pair"]),
                "edges_count": int(feat["edges_count"]),
                "angle_deg": float(feat["angle_deg"]),
                "angle_to_red_signed_deg": float(feat["angle_to_red_signed_deg"]),
                "bottom_intersect_x_norm": float(feat["bottom_intersect_x_norm"]),
                "mid_intersect_x_norm": float(feat["mid_intersect_x_norm"]),
                "center_offset_x_norm": float(feat["center_offset_x_norm"]),
                "bottom_intersect_dx_px": float(feat["bottom_intersect_dx_px"]),
                "mid_intersect_dx_px": float(feat["mid_intersect_dx_px"]),
                "center_offset_x_px": float(feat["center_offset_x_px"]),
            }

        info["lane1"] = pack_center(lane1, 1)
        info["lane2"] = pack_center(lane2, 2)
        info["selected"] = pack_center(selected_lane, int(selected_lane_id) if selected_lane_id else 0)

        for d in center_sorted:
            info["lanes_centerline"].append({
                "confidence": float(d["confidence"]),
                "has_pair": bool(d["has_pair"]),
                "edges_count": int(d["edges_count"]),
                "angle_deg": float(d["angle_deg"]),
                "angle_to_red_signed_deg": float(d["angle_to_red_signed_deg"]),
                "bottom_intersect_x_norm": float(d["bottom_intersect_x_norm"]),
                "mid_intersect_x_norm": float(d["mid_intersect_x_norm"]),
                "center_offset_x_norm": float(d["center_offset_x_norm"]),
                "bottom_intersect_dx_px": float(d["bottom_intersect_dx_px"]),
                "mid_intersect_dx_px": float(d["mid_intersect_dx_px"]),
                "center_offset_x_px": float(d["center_offset_x_px"]),
                "angle_to_vertical_abs_err": float(d["angle_to_vertical_abs_err"]),
            })

        # single-edge hints
        for e in single_edges:
            info["lanes_single_edge"].append({
                "confidence": float(e["confidence"]),
                "edge_bottom_dx_norm": _norm(e["edge_bottom_dx_norm"]),
                "edge_bottom_dx_px": float(e["edge_bottom_dx_px"]),
                "search_dir": str(e["search_dir"]),
                "estimated_center_dx_norm": _norm(e["estimated_center_dx_norm"]),
                "estimated_center_dx_px": float(e["estimated_center_dx_px"]),
                "angle_deg": float(e["angle_deg"]),
                "angle_to_red_signed_deg": float(e["angle_to_red_signed_deg"]),
            })

        m = String()
        m.data = json.dumps(info, separators=(",", ":"))
        self.pub_info.publish(m)

        # fps log
        self._n_fps += 1
        now = time.time()
        if now - self._t_fps >= 2.0:
            fps = self._n_fps / (now - self._t_fps)
            self.get_logger().info(f"lane_centerline_node fps ~ {fps:.2f}")
            self._t_fps = now
            self._n_fps = 0


def main():
    rclpy.init()
    node = LaneCenterlineNode320x240()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()