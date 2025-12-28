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

# =========================================================
# FIXED RESOLUTION (ONLY 320x240)
# =========================================================
W_TARGET = 320
H_TARGET = 240

# =========================================================
# DEFAULT SETTINGS (✅ 320x240 tuned)
# =========================================================
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
DRAW_LINE_TYPE = cv2.LINE_8         # CPU friendly


# =========================================================
# LINE UTILS
# =========================================================
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
            c, edge255.shape[1], edge255.shape[0],
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


def angle_to_vertical_signed_deg(x1, y1, x2, y2) -> float:
    """
    Signed angle relative to vertical red line.
    Image coords: +x right, +y down.
    We define:
      0 deg = vertical (up/down)
      + = leaning to the right
      - = leaning to the left
    """
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    ang = np.degrees(np.arctan2(dx, dy))  # atan2(x, y) => relative to vertical axis
    # wrap to [-90, +90] roughly
    if ang > 90.0:
        ang -= 180.0
    if ang < -90.0:
        ang += 180.0
    return float(ang)


def process_one(mask255: np.ndarray, params: dict, cache: PreprocCache, vis: np.ndarray):
    edge = cache.filled_to_outline_255(mask255)
    cache.erase_border_inplace(edge, params["border_erase"])

    h, w = edge.shape
    vis.fill(0)

    structs = edge_to_structures(edge, params)

    # center red vertical
    cx = w // 2
    cv2.line(vis, (cx, 0), (cx, h - 1), (0, 0, 255), 1, DRAW_LINE_TYPE)

    # sort by closeness to vertical (90deg in your cluster angle space)
    structs = sorted(structs, key=lambda D: _ang_dist(D["angle"], 90.0))

    # draw
    for idx, D in enumerate(structs):
        for (x1, y1, x2, y2) in D["edges"]:
            ext = extend_to_borders(x1, y1, x2, y2, w, h)
            if ext is None:
                continue
            ex1, ey1, ex2, ey2 = ext
            cv2.line(vis, (ex1, ey1), (ex2, ey2), LANE_EDGE_COLOR, LANE_EDGE_THICKNESS, DRAW_LINE_TYPE)

        if D["center"] is not None:
            (x1, y1, x2, y2) = D["center"]
            ext = extend_to_borders(x1, y1, x2, y2, w, h)
            if ext is None:
                continue
            ex1, ey1, ex2, ey2 = ext
            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
            cv2.line(vis, (ex1, ey1), (ex2, ey2), color, params["center_thickness"], DRAW_LINE_TYPE)

    return edge, vis, structs


class LaneCenterlineNode320x240(Node):
    """
    ONLY WORKS AT 320x240.

    Sub:  /camera_bottom/lane_mask        (mono8 0/255)  -> MUST be 320x240
    Pub:  /camera_bottom/center_lines     (bgr8)         -> 320x240 visualization
    Pub:  /camera_bottom/lane_info        (std_msgs/String JSON)
    """
    def __init__(self):
        super().__init__("lane_centerline_node")

        self.declare_parameter("in_topic", "/camera_bottom/lane_mask")
        self.declare_parameter("out_topic", "/camera_bottom/center_lines")
        self.declare_parameter("info_topic", "/camera_bottom/lane_info")
        self.declare_parameter("skip_n", 0)

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)
        self.skip_n = int(self.get_parameter("skip_n").value)

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

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(Image, self.in_topic, self.cb, qos)
        self.pub_img = self.create_publisher(Image, self.out_topic, 1)
        self.pub_info = self.create_publisher(String, self.info_topic, 1)

        self._i = 0
        self._t_last = time.time()
        self._count = 0

        self._vis = np.zeros((H_TARGET, W_TARGET, 3), np.uint8)

        self.get_logger().info("READY (ONLY 320x240) + lane_info JSON")
        self.get_logger().info(f"Sub : {self.in_topic}")
        self.get_logger().info(f"Pub : {self.out_topic}")
        self.get_logger().info(f"Info: {self.info_topic}")

    def cb(self, msg: Image):
        self._i += 1
        if self.skip_n > 0 and (self._i % (self.skip_n + 1)) != 0:
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        # STRICT: must be 320x240
        if mask.shape[0] != H_TARGET or mask.shape[1] != W_TARGET:
            self.get_logger().error(
                f"Mask must be {W_TARGET}x{H_TARGET}, got {mask.shape[1]}x{mask.shape[0]}"
            )
            return

        _, vis, structs = process_one(mask, self.params, self.cache, self._vis)

        # publish visualization
        out_img = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_img.header = msg.header
        self.pub_img.publish(out_img)

        # ----------------------------
        # build + publish info JSON
        # ----------------------------
        w, h = W_TARGET, H_TARGET
        cx = w // 2

        total_orange_edges = sum(len(s["edges"]) for s in structs)
        num_pairs = sum(1 for s in structs if len(s["edges"]) == 2)
        num_singles = sum(1 for s in structs if len(s["edges"]) == 1)

        info = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "img": {"w": w, "h": h, "center_x": cx},
            "num_structs": int(len(structs)),
            "total_orange_edges": int(total_orange_edges),
            "num_pairs": int(num_pairs),
            "num_single_edges": int(num_singles),
            "best_struct_index": 0 if len(structs) > 0 else -1,
            "structs": []
        }

        for idx, s in enumerate(structs):
            edges_count = int(len(s["edges"]))
            has_pair = (edges_count == 2)
            center_present = (s["center"] is not None)

            st = {
                "index": int(idx),
                "angle_deg": float(s["angle"]),   # cluster angle 0..180
                "edges_count": edges_count,
                "has_pair": bool(has_pair),
                "centerline_present": bool(center_present),
            }

            if center_present:
                x1, y1, x2, y2 = s["center"]
                # use clipped line for stability
                ext = extend_to_borders(x1, y1, x2, y2, w, h)
                if ext is not None:
                    x1, y1, x2, y2 = ext

                mid_x = 0.5 * (x1 + x2)
                mid_y = 0.5 * (y1 + y2)

                ang_signed = angle_to_vertical_signed_deg(x1, y1, x2, y2)
                ang_abs = abs(ang_signed)

                offset_px = float(mid_x - cx)
                offset_norm = float(offset_px / (w * 0.5 + 1e-9))  # ~[-1,+1]

                st.update({
                    "angle_to_red_deg": float(ang_abs),
                    "angle_to_red_signed_deg": float(ang_signed),
                    "center_mid": {"x": float(mid_x), "y": float(mid_y)},
                    "center_offset_x_px": float(offset_px),
                    "center_offset_x_norm": float(offset_norm),
                })
            else:
                # still provide an angle estimate from the strongest edge if you want
                st.update({
                    "angle_to_red_deg": None,
                    "angle_to_red_signed_deg": None,
                    "center_mid": None,
                    "center_offset_x_px": None,
                    "center_offset_x_norm": None,
                })

            info["structs"].append(st)

        msg_info = String()
        msg_info.data = json.dumps(info, separators=(",", ":"))
        self.pub_info.publish(msg_info)

        # fps log
        self._count += 1
        now = time.time()
        if now - self._t_last >= 2.0:
            fps = self._count / (now - self._t_last)
            self.get_logger().info(f"centerline fps ~ {fps:.2f}")
            self._t_last = now
            self._count = 0


def main():
    rclpy.init()
    node = LaneCenterlineNode320x240()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()