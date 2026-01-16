#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

namespace {
constexpr int W_TARGET = 320;
constexpr int H_TARGET = 240;

constexpr int OUTLINE_K = 5;
constexpr int EDGE_THICKNESS = 2;
constexpr int BORDER_ERASE = 3;

constexpr int HOUGH_THRESH = 22;
constexpr int HOUGH_MIN_LEN = 40;
constexpr int HOUGH_MAX_GAP = 10;
constexpr float ANGLE_MERGE_DEG = 12.0f;

constexpr float MIN_PARALLEL_SEP = 18.0f;
constexpr float MAX_PARALLEL_SEP = 120.0f;
constexpr float MIN_GROUP_LEN = 40.0f;
constexpr float MIN_T_OVERLAP = 0.20f;

constexpr bool DRAW_SINGLE_EDGE_CANDIDATE = true;
constexpr float SINGLE_EDGE_MIN_TOTAL_LEN = 90.0f;
constexpr float SINGLE_EDGE_MIN_SPAN_T = 70.0f;

constexpr int MAX_CENTERLINES = 2;
constexpr int CENTER_THICKNESS = 2;

constexpr int DRAW_LINE_TYPE = cv::LINE_8;

const cv::Scalar LANE_EDGE_COLOR(0, 140, 255);
const cv::Scalar COLOR_SELECTED(0, 255, 0);
const cv::Scalar COLOR_OTHER(255, 0, 0);
const cv::Scalar SELECTED_POINT_COLOR(0, 0, 255);
constexpr int SELECTED_POINT_RADIUS = 5;

inline float clampf(float x, float lo, float hi) {
  return (x < lo) ? lo : (x > hi ? hi : x);
}

std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\\': o << "\\\\"; break;
      case '"': o << "\\\""; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default: o << c; break;
    }
  }
  return o.str();
}

std::optional<cv::Vec4i> extend_to_borders(int x1, int y1, int x2, int y2, int w, int h, int scale = 10000) {
  int dx = x2 - x1;
  int dy = y2 - y1;
  if (dx == 0 && dy == 0) {
    return std::nullopt;
  }
  int xa = static_cast<int>(std::lround(x1 - dx * scale));
  int ya = static_cast<int>(std::lround(y1 - dy * scale));
  int xb = static_cast<int>(std::lround(x1 + dx * scale));
  int yb = static_cast<int>(std::lround(y1 + dy * scale));

  cv::Point p1(xa, ya);
  cv::Point p2(xb, yb);
  cv::Rect rect(0, 0, w, h);
  bool ok = cv::clipLine(rect, p1, p2);
  if (!ok) {
    return std::nullopt;
  }
  return cv::Vec4i(p1.x, p1.y, p2.x, p2.y);
}

float angle_deg(int x1, int y1, int x2, int y2) {
  float ang = static_cast<float>(std::atan2((y2 - y1), (x2 - x1)) * 180.0 / M_PI);
  ang = std::fmod(ang + 180.0f, 180.0f);
  return ang;
}

float ang_dist(float a, float b) {
  float d = std::fabs(a - b);
  return std::min(d, 180.0f - d);
}

float angle_to_vertical_signed_deg(int x1, int y1, int x2, int y2) {
  float dx = static_cast<float>(x2 - x1);
  float dy = static_cast<float>(y2 - y1);
  if (std::fabs(dx) < 1e-6f && std::fabs(dy) < 1e-6f) {
    return 0.0f;
  }
  float ang = static_cast<float>(std::atan2(dx, dy) * 180.0 / M_PI);
  if (ang > 90.0f) {
    ang -= 180.0f;
  }
  if (ang < -90.0f) {
    ang += 180.0f;
  }
  return ang;
}

std::optional<float> line_x_at_y(int x1, int y1, int x2, int y2, int y_query) {
  float y1f = static_cast<float>(y1);
  float y2f = static_cast<float>(y2);
  if (std::fabs(y2f - y1f) < 1e-6f) {
    return std::nullopt;
  }
  float t = (static_cast<float>(y_query) - y1f) / (y2f - y1f);
  return static_cast<float>(x1) + t * static_cast<float>(x2 - x1);
}

struct LineSeg {
  int x1;
  int y1;
  int x2;
  int y2;
  float ang;
  float len;
};

struct Cluster {
  float mean;
  float score;
  std::vector<LineSeg> items;
};

float robust_mean_angle(const std::vector<LineSeg>& items) {
  double s = 0.0;
  double c = 0.0;
  for (const auto& it : items) {
    double ang = it.ang * M_PI / 180.0;
    double w = static_cast<double>(it.len);
    c += std::cos(2.0 * ang) * w;
    s += std::sin(2.0 * ang) * w;
  }
  double mean = 0.5 * std::atan2(s, c) * 180.0 / M_PI;
  mean = std::fmod(mean + 180.0, 180.0);
  return static_cast<float>(mean);
}

std::vector<Cluster> cluster_by_angle(const std::vector<LineSeg>& lines, float tol_deg) {
  std::vector<Cluster> clusters;
  for (const auto& L : lines) {
    bool placed = false;
    for (auto& cl : clusters) {
      if (ang_dist(L.ang, cl.mean) <= tol_deg) {
        cl.items.push_back(L);
        cl.mean = robust_mean_angle(cl.items);
        placed = true;
        break;
      }
    }
    if (!placed) {
      Cluster cl;
      cl.mean = L.ang;
      cl.items = {L};
      cl.score = 0.0f;
      clusters.push_back(cl);
    }
  }

  for (auto& cl : clusters) {
    cl.mean = robust_mean_angle(cl.items);
    float sum = 0.0f;
    for (const auto& it : cl.items) {
      sum += it.len;
    }
    cl.score = sum;
  }

  std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
    return a.score > b.score;
  });
  return clusters;
}

struct LineFeat {
  float angle_deg;
  float angle_to_red_signed_deg;
  float bottom_intersect_x_norm;
  float mid_intersect_x_norm;
  float center_offset_x_norm;
  float bottom_intersect_dx_px;
  float mid_intersect_dx_px;
  float center_offset_x_px;
  cv::Vec4i line;
  float angle_to_vertical_abs_err;
};

struct LaneStruct {
  float angle;
  std::optional<cv::Vec4i> center;
  std::vector<cv::Vec4i> edges;
};

struct Params {
  int border_erase;
  int hough_thresh;
  int hough_min_len;
  int hough_max_gap;
  float angle_merge_deg;
  float min_parallel_sep;
  float max_parallel_sep;
  float min_group_len;
  float min_t_overlap;
  bool draw_single_edge;
  float single_edge_min_total_len;
  float single_edge_min_span_t;
  int max_centerlines;
  int center_thickness;
};

struct SingleEdgeInfo {
  float confidence;
  float edge_bottom_dx_norm;
  float edge_bottom_dx_px;
  std::string search_dir;
  float estimated_center_dx_norm;
  float estimated_center_dx_px;
  float angle_deg;
  float angle_to_red_signed_deg;
};

struct SelectedLineCenter {
  float x_px;
  float y_px;
  float dx_px;
  float dx_norm;
};

std::optional<cv::Vec4i> line_from_u_c(float ux, float uy, float nx, float ny, float c_val,
                                       float tmin, float tmax, int w, int h) {
  float xA = ux * tmin + nx * c_val;
  float yA = uy * tmin + ny * c_val;
  float xB = ux * tmax + nx * c_val;
  float yB = uy * tmax + ny * c_val;

  int x1 = static_cast<int>(std::lround(xA));
  int y1 = static_cast<int>(std::lround(yA));
  int x2 = static_cast<int>(std::lround(xB));
  int y2 = static_cast<int>(std::lround(yB));

  cv::Point p1(x1, y1);
  cv::Point p2(x2, y2);
  cv::Rect rect(0, 0, w, h);
  if (!cv::clipLine(rect, p1, p2)) {
    return std::nullopt;
  }
  return cv::Vec4i(p1.x, p1.y, p2.x, p2.y);
}

std::optional<LaneStruct> centerline_and_edges_from_cluster(
    const Cluster& cluster, int w, int h,
    float min_parallel_sep, float max_parallel_sep, float min_group_len, float min_t_overlap,
    bool draw_single_edge, float single_edge_min_total_len, float single_edge_min_span_t) {

  float theta = cluster.mean * M_PI / 180.0f;
  float ux = std::cos(theta);
  float uy = std::sin(theta);
  float nx = -uy;
  float ny = ux;

  const auto& items = cluster.items;
  if (items.size() < 2) {
    return std::nullopt;
  }

  struct Seg {
    int x1, y1, x2, y2;
    float len;
    float c;
  };

  std::vector<Seg> segs;
  std::vector<cv::Point2f> pts_all;
  segs.reserve(items.size());
  for (const auto& it : items) {
    float mx = 0.5f * (it.x1 + it.x2);
    float my = 0.5f * (it.y1 + it.y2);
    float c = nx * mx + ny * my;
    segs.push_back({it.x1, it.y1, it.x2, it.y2, it.len, c});
    pts_all.emplace_back(it.x1, it.y1);
    pts_all.emplace_back(it.x2, it.y2);
  }

  if (segs.size() < 2) {
    return std::nullopt;
  }

  float total_len = 0.0f;
  for (const auto& s : segs) {
    total_len += s.len;
  }

  float tmin_all = 1e9f;
  float tmax_all = -1e9f;
  for (const auto& p : pts_all) {
    float t = p.x * ux + p.y * uy;
    tmin_all = std::min(tmin_all, t);
    tmax_all = std::max(tmax_all, t);
  }
  float span_t_all = tmax_all - tmin_all;

  float cmin = 1e9f;
  float cmax = -1e9f;
  for (const auto& s : segs) {
    cmin = std::min(cmin, s.c);
    cmax = std::max(cmax, s.c);
  }
  float c_spread = cmax - cmin;

  bool two_side = false;
  std::optional<cv::Vec4i> center_line;
  std::vector<cv::Vec4i> edge_lines;

  if (c_spread >= min_parallel_sep) {
    std::vector<float> c_vals;
    c_vals.reserve(segs.size());
    for (const auto& s : segs) {
      c_vals.push_back(s.c);
    }
    std::sort(c_vals.begin(), c_vals.end());

    float mu1 = c_vals[static_cast<size_t>(0.25f * (c_vals.size() - 1))];
    float mu2 = c_vals[static_cast<size_t>(0.75f * (c_vals.size() - 1))];

    std::vector<int> lab;
    for (int iter = 0; iter < 10; ++iter) {
      lab.resize(c_vals.size());
      for (size_t i = 0; i < c_vals.size(); ++i) {
        float d1 = std::fabs(c_vals[i] - mu1);
        float d2 = std::fabs(c_vals[i] - mu2);
        lab[i] = (d2 < d1) ? 1 : 0;
      }

      bool all0 = std::all_of(lab.begin(), lab.end(), [](int v){ return v == 0; });
      bool all1 = std::all_of(lab.begin(), lab.end(), [](int v){ return v == 1; });
      if (all0 || all1) {
        break;
      }

      float sum1 = 0.0f, sum2 = 0.0f;
      int n1 = 0, n2 = 0;
      for (size_t i = 0; i < c_vals.size(); ++i) {
        if (lab[i] == 0) { sum1 += c_vals[i]; n1++; }
        else { sum2 += c_vals[i]; n2++; }
      }
      float mu1_new = (n1 > 0) ? sum1 / n1 : mu1;
      float mu2_new = (n2 > 0) ? sum2 / n2 : mu2;
      if (std::fabs(mu1_new - mu1) < 1e-3f && std::fabs(mu2_new - mu2) < 1e-3f) {
        mu1 = mu1_new;
        mu2 = mu2_new;
        break;
      }
      mu1 = mu1_new;
      mu2 = mu2_new;
    }

    std::vector<Seg> g1, g2;
    for (const auto& s : segs) {
      float d1 = std::fabs(s.c - mu1);
      float d2 = std::fabs(s.c - mu2);
      if (d2 < d1) {
        g2.push_back(s);
      } else {
        g1.push_back(s);
      }
    }

    float len1 = 0.0f, len2 = 0.0f;
    for (const auto& s : g1) len1 += s.len;
    for (const auto& s : g2) len2 += s.len;

    float sep = std::fabs(mu2 - mu1);
    if (len1 >= min_group_len && len2 >= min_group_len && sep >= min_parallel_sep && sep <= max_parallel_sep) {
      float c1 = 0.0f, c2 = 0.0f;
      for (const auto& s : g1) c1 += s.c * s.len;
      for (const auto& s : g2) c2 += s.c * s.len;
      c1 /= (len1 + 1e-9f);
      c2 /= (len2 + 1e-9f);
      float c_mid = 0.5f * (c1 + c2);

      auto t_range = [&](const std::vector<Seg>& group) {
        float tmin = 1e9f;
        float tmax = -1e9f;
        for (const auto& s : group) {
          float t1 = s.x1 * ux + s.y1 * uy;
          float t2 = s.x2 * ux + s.y2 * uy;
          tmin = std::min(tmin, std::min(t1, t2));
          tmax = std::max(tmax, std::max(t1, t2));
        }
        return std::pair<float, float>(tmin, tmax);
      };

      auto r1 = t_range(g1);
      auto r2 = t_range(g2);
      float ov = std::max(0.0f, std::min(r1.second, r2.second) - std::max(r1.first, r2.first));
      float span = std::max(1e-6f, std::max(r1.second, r2.second) - std::min(r1.first, r2.first));
      if ((ov / span) >= min_t_overlap) {
        float tmin = std::min(r1.first, r2.first);
        float tmax = std::max(r1.second, r2.second);
        auto e1 = line_from_u_c(ux, uy, nx, ny, c1, tmin, tmax, w, h);
        auto e2 = line_from_u_c(ux, uy, nx, ny, c2, tmin, tmax, w, h);
        auto cc = line_from_u_c(ux, uy, nx, ny, c_mid, tmin, tmax, w, h);
        if (e1 && e2 && cc) {
          two_side = true;
          edge_lines = {*e1, *e2};
          center_line = *cc;
        }
      }
    }
  }

  if (!two_side && draw_single_edge) {
    if (total_len >= single_edge_min_total_len && span_t_all >= single_edge_min_span_t) {
      float c_hat = 0.0f;
      for (const auto& s : segs) {
        c_hat += s.c * s.len;
      }
      c_hat /= (total_len + 1e-9f);
      auto e = line_from_u_c(ux, uy, nx, ny, c_hat, tmin_all, tmax_all, w, h);
      if (e) {
        edge_lines = {*e};
        center_line = std::nullopt;
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }

  if (edge_lines.empty() && !center_line.has_value()) {
    return std::nullopt;
  }

  LaneStruct out;
  out.angle = cluster.mean;
  out.center = center_line;
  out.edges = edge_lines;
  return out;
}

std::vector<LaneStruct> edge_to_structures(const cv::Mat& edge255, const Params& params) {
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(edge255, linesP, 1, CV_PI / 180.0, params.hough_thresh,
                  params.hough_min_len, params.hough_max_gap);

  if (linesP.empty()) {
    return {};
  }

  std::vector<LineSeg> lines;
  lines.reserve(linesP.size());
  for (const auto& l : linesP) {
    int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    float L = std::hypot(static_cast<float>(x2 - x1), static_cast<float>(y2 - y1));
    if (L < 10.0f) {
      continue;
    }
    float ang = angle_deg(x1, y1, x2, y2);
    lines.push_back({x1, y1, x2, y2, ang, L});
  }

  if (lines.empty()) {
    return {};
  }

  auto clusters = cluster_by_angle(lines, params.angle_merge_deg);

  std::vector<LaneStruct> structs;
  for (const auto& c : clusters) {
    auto st = centerline_and_edges_from_cluster(
        c, edge255.cols, edge255.rows,
        params.min_parallel_sep, params.max_parallel_sep,
        params.min_group_len, params.min_t_overlap,
        params.draw_single_edge, params.single_edge_min_total_len,
        params.single_edge_min_span_t);
    if (!st) {
      continue;
    }

    if (!st->center.has_value() && st->edges.size() == 2) {
      auto cc = extend_to_borders(st->edges[0][0], st->edges[0][1], st->edges[0][2], st->edges[0][3],
                                  edge255.cols, edge255.rows);
      auto dd = extend_to_borders(st->edges[1][0], st->edges[1][1], st->edges[1][2], st->edges[1][3],
                                  edge255.cols, edge255.rows);
      if (cc && dd) {
        cv::Vec4i e1 = *cc;
        cv::Vec4i e2 = *dd;

        int x1a = e1[0], y1a = e1[1], x1b = e1[2], y1b = e1[3];
        int x2a = e2[0], y2a = e2[1], x2b = e2[2], y2b = e2[3];

        if (y1a > y1b) std::swap(x1a, x1b), std::swap(y1a, y1b);
        if (y2a > y2b) std::swap(x2a, x2b), std::swap(y2a, y2b);

        int cx_a = static_cast<int>(std::lround(0.5 * (x1a + x2a)));
        int cy_a = static_cast<int>(std::lround(0.5 * (y1a + y2a)));
        int cx_b = static_cast<int>(std::lround(0.5 * (x1b + x2b)));
        int cy_b = static_cast<int>(std::lround(0.5 * (y1b + y2b)));

        cv::Point p1(cx_a, cy_a);
        cv::Point p2(cx_b, cy_b);
        cv::Rect rect(0, 0, edge255.cols, edge255.rows);
        if (cv::clipLine(rect, p1, p2)) {
          st->center = cv::Vec4i(p1.x, p1.y, p2.x, p2.y);
        }
      }
    }

    bool ok = true;
    for (const auto& ex : structs) {
      if (ang_dist(st->angle, ex.angle) < 18.0f) {
        ok = false;
        break;
      }
    }
    if (ok) {
      structs.push_back(*st);
    }
    if (static_cast<int>(structs.size()) >= params.max_centerlines) {
      break;
    }
  }

  return structs;
}

float compute_confidence(const LaneStruct& s) {
  int edges_count = static_cast<int>(s.edges.size());
  bool has_pair = (edges_count == 2);
  bool center_present = s.center.has_value();

  float ang = s.angle;
  float ang_err = ang_dist(ang, 90.0f);
  float angle_score = std::max(0.0f, 1.0f - (ang_err / 45.0f));

  float score = 0.0f;
  score += has_pair ? 0.55f : 0.15f;
  score += center_present ? 0.25f : 0.0f;
  score += 0.20f * angle_score;
  return clampf(score, 0.0f, 1.0f);
}

std::optional<LineFeat> feat_from_line(const cv::Vec4i& line, int w, int h) {
  int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
  auto ext = extend_to_borders(x1, y1, x2, y2, w, h);
  if (ext) {
    x1 = (*ext)[0];
    y1 = (*ext)[1];
    x2 = (*ext)[2];
    y2 = (*ext)[3];
  }

  float ang_deg = angle_deg(x1, y1, x2, y2);
  float ang_to_vert = angle_to_vertical_signed_deg(x1, y1, x2, y2);

  int cx = w / 2;
  int y_bottom = h - 1;
  int y_mid = h / 2;

  auto xb_opt = line_x_at_y(x1, y1, x2, y2, y_bottom);
  auto xm_opt = line_x_at_y(x1, y1, x2, y2, y_mid);
  float xb = xb_opt.has_value() ? *xb_opt : 0.5f * (x1 + x2);
  float xm = xm_opt.has_value() ? *xm_opt : xb;

  xb = clampf(xb, 0.0f, static_cast<float>(w - 1));
  xm = clampf(xm, 0.0f, static_cast<float>(w - 1));

  float bottom_dx_px = xb - static_cast<float>(cx);
  float mid_dx_px = xm - static_cast<float>(cx);

  float bottom_dx_norm = bottom_dx_px / (w * 0.5f + 1e-9f);
  float mid_dx_norm = mid_dx_px / (w * 0.5f + 1e-9f);

  float mid_x_seg = 0.5f * (x1 + x2);
  float center_offset_px = mid_x_seg - static_cast<float>(cx);
  float center_offset_norm = center_offset_px / (w * 0.5f + 1e-9f);

  LineFeat f;
  f.angle_deg = ang_deg;
  f.angle_to_red_signed_deg = ang_to_vert;
  f.bottom_intersect_x_norm = bottom_dx_norm;
  f.mid_intersect_x_norm = mid_dx_norm;
  f.center_offset_x_norm = center_offset_norm;
  f.bottom_intersect_dx_px = bottom_dx_px;
  f.mid_intersect_dx_px = mid_dx_px;
  f.center_offset_x_px = center_offset_px;
  f.line = cv::Vec4i(x1, y1, x2, y2);
  f.angle_to_vertical_abs_err = ang_dist(ang_deg, 90.0f);
  return f;
}

std::string pack_center_json(const std::optional<LineFeat>& feat, int lane_id, float confidence, bool has_pair, int edges_count) {
  if (!feat.has_value()) {
    return "null";
  }
  const auto& f = *feat;
  std::ostringstream ss;
  ss << std::setprecision(6);
  ss << "{"
     << "\"id\":" << lane_id << ","
     << "\"confidence\":" << confidence << ","
     << "\"has_pair\":" << (has_pair ? "true" : "false") << ","
     << "\"edges_count\":" << edges_count << ","
     << "\"angle_deg\":" << f.angle_deg << ","
     << "\"angle_to_red_signed_deg\":" << f.angle_to_red_signed_deg << ","
     << "\"bottom_intersect_x_norm\":" << f.bottom_intersect_x_norm << ","
     << "\"mid_intersect_x_norm\":" << f.mid_intersect_x_norm << ","
     << "\"center_offset_x_norm\":" << f.center_offset_x_norm << ","
     << "\"bottom_intersect_dx_px\":" << f.bottom_intersect_dx_px << ","
     << "\"mid_intersect_dx_px\":" << f.mid_intersect_dx_px << ","
     << "\"center_offset_x_px\":" << f.center_offset_x_px
     << "}";
  return ss.str();
}

}  // namespace

class LaneCenterlineNode : public rclcpp::Node {
 public:
  LaneCenterlineNode()
  : rclcpp::Node("lane_centerline_node_cpp") {
    declare_parameter("in_topic", "/camera_bottom/lane_mask");
    declare_parameter("out_topic", "/camera_bottom/center_lines");
    declare_parameter("info_topic", "/camera_bottom/lane_info");
    declare_parameter("change_lane_topic", "/camera_bottom/change_lane");
    declare_parameter("skip_n", 0);
    declare_parameter("lane_half_width_px", 60.0);

    in_topic_ = get_parameter("in_topic").as_string();
    out_topic_ = get_parameter("out_topic").as_string();
    info_topic_ = get_parameter("info_topic").as_string();
    change_lane_topic_ = get_parameter("change_lane_topic").as_string();
    skip_n_ = get_parameter("skip_n").as_int();
    lane_half_width_px_ = static_cast<float>(get_parameter("lane_half_width_px").as_double());

    params_ = {
      BORDER_ERASE,
      HOUGH_THRESH,
      HOUGH_MIN_LEN,
      HOUGH_MAX_GAP,
      static_cast<float>(ANGLE_MERGE_DEG),
      static_cast<float>(MIN_PARALLEL_SEP),
      static_cast<float>(MAX_PARALLEL_SEP),
      static_cast<float>(MIN_GROUP_LEN),
      static_cast<float>(MIN_T_OVERLAP),
      static_cast<bool>(DRAW_SINGLE_EDGE_CANDIDATE),
      static_cast<float>(SINGLE_EDGE_MIN_TOTAL_LEN),
      static_cast<float>(SINGLE_EDGE_MIN_SPAN_T),
      static_cast<int>(MAX_CENTERLINES),
      static_cast<int>(CENTER_THICKNESS),
    };

    cache_k_grad_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(OUTLINE_K, OUTLINE_K));
    if (EDGE_THICKNESS > 1) {
      cache_k_thick_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(EDGE_THICKNESS, EDGE_THICKNESS));
    }

    vis_ = cv::Mat(H_TARGET, W_TARGET, CV_8UC3, cv::Scalar(0, 0, 0));

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        in_topic_, qos, std::bind(&LaneCenterlineNode::cb, this, std::placeholders::_1));
    pub_img_ = create_publisher<sensor_msgs::msg::Image>(out_topic_, qos);
    pub_info_ = create_publisher<std_msgs::msg::String>(info_topic_, qos);
    sub_change_ = create_subscription<std_msgs::msg::String>(
        change_lane_topic_, qos, std::bind(&LaneCenterlineNode::cb_change_lane, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "READY lane_centerline_node_cpp");
    RCLCPP_INFO(get_logger(), "Sub  : %s", in_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Pub  : %s", out_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Info : %s", info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "ChangeLane : %s", change_lane_topic_.c_str());
  }

 private:
  void cb_change_lane(const std_msgs::msg::String::SharedPtr msg) {
    std::string s = msg->data;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "1" || s == "lane1") {
      selected_lane_id_ = 1;
      RCLCPP_INFO(get_logger(), "change_lane -> selected lane1");
    } else if (s == "2" || s == "lane2") {
      selected_lane_id_ = 2;
      RCLCPP_INFO(get_logger(), "change_lane -> selected lane2");
    } else if (s == "toggle" || s == "switch") {
      selected_lane_id_ = (selected_lane_id_ == 1) ? 2 : 1;
      RCLCPP_INFO(get_logger(), "change_lane -> toggled to lane%d", selected_lane_id_);
    } else {
      RCLCPP_WARN(get_logger(), "change_lane ignored: '%s'", msg->data.c_str());
    }
  }

  cv::Mat filled_to_outline_255(const cv::Mat& mask) {
    cv::Mat grad;
    cv::morphologyEx(mask, grad, cv::MORPH_GRADIENT, cache_k_grad_);
    cv::Mat out;
    cv::compare(grad, 0, out, cv::CMP_GT);
    out.convertTo(out, CV_8U, 255);
    if (!cache_k_thick_.empty()) {
      cv::dilate(out, out, cache_k_thick_);
    }
    return out;
  }

  void erase_border_inplace(cv::Mat& bin255, int px) {
    if (px <= 0) return;
    px = std::max(0, std::min(px, std::min(bin255.rows, bin255.cols) / 2));
    bin255.rowRange(0, px).setTo(0);
    bin255.rowRange(bin255.rows - px, bin255.rows).setTo(0);
    bin255.colRange(0, px).setTo(0);
    bin255.colRange(bin255.cols - px, bin255.cols).setTo(0);
  }

  void cb(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (skip_n_ > 0 && (frame_i_++ % (skip_n_ + 1)) != 0) {
      return;
    }

    const std::string enc = msg->encoding;
    if (enc != "mono8" && enc != "8UC1") {
      RCLCPP_ERROR(get_logger(), "Expected mono8/8UC1 image, got encoding='%s'", enc.c_str());
      return;
    }

    if (static_cast<int>(msg->height) != H_TARGET || static_cast<int>(msg->width) != W_TARGET) {
      RCLCPP_ERROR(get_logger(), "Mask must be %dx%d, got %dx%d",
                   W_TARGET, H_TARGET, msg->width, msg->height);
      return;
    }

    if (msg->data.empty()) {
      RCLCPP_ERROR(get_logger(), "Empty image data");
      return;
    }

    cv::Mat mask_view(
        static_cast<int>(msg->height),
        static_cast<int>(msg->width),
        CV_8UC1,
        const_cast<unsigned char*>(msg->data.data()),
        static_cast<size_t>(msg->step)
    );
    cv::Mat mask = mask_view.clone();

    cv::Mat edge = filled_to_outline_255(mask);
    erase_border_inplace(edge, params_.border_erase);

    std::vector<LaneStruct> structs = edge_to_structures(edge, params_);

    vis_.setTo(cv::Scalar(0, 0, 0));
    int cx = W_TARGET / 2;
    cv::line(vis_, cv::Point(cx, 0), cv::Point(cx, H_TARGET - 1), cv::Scalar(0, 0, 255), 1, DRAW_LINE_TYPE);

    for (const auto& D : structs) {
      for (const auto& e : D.edges) {
        auto ext = extend_to_borders(e[0], e[1], e[2], e[3], W_TARGET, H_TARGET);
        if (!ext) continue;
        cv::line(vis_, cv::Point((*ext)[0], (*ext)[1]), cv::Point((*ext)[2], (*ext)[3]),
                 LANE_EDGE_COLOR, EDGE_THICKNESS, DRAW_LINE_TYPE);
      }
    }

    std::vector<std::pair<LineFeat, LaneStruct>> center_dets;
    std::vector<SingleEdgeInfo> single_edges;

    for (const auto& s : structs) {
      if (s.center.has_value()) {
        auto feat = feat_from_line(*s.center, W_TARGET, H_TARGET);
        if (feat.has_value()) {
          center_dets.emplace_back(*feat, s);
        }
      } else if (s.edges.size() == 1) {
        auto feat = feat_from_line(s.edges[0], W_TARGET, H_TARGET);
        if (feat.has_value()) {
          float edge_bottom_dx_px = feat->bottom_intersect_dx_px;
          float edge_bottom_dx_norm = feat->bottom_intersect_x_norm;
          std::string search_dir = (edge_bottom_dx_px < 0.0f) ? "right" : "left";
          float est_center_dx_px = (edge_bottom_dx_px < 0.0f)
              ? (edge_bottom_dx_px + lane_half_width_px_)
              : (edge_bottom_dx_px - lane_half_width_px_);
          float est_center_dx_norm = est_center_dx_px / (W_TARGET * 0.5f + 1e-9f);

          SingleEdgeInfo e;
          e.confidence = compute_confidence(s);
          e.edge_bottom_dx_norm = clampf(edge_bottom_dx_norm, -1.0f, 1.0f);
          e.edge_bottom_dx_px = edge_bottom_dx_px;
          e.search_dir = search_dir;
          e.estimated_center_dx_norm = clampf(est_center_dx_norm, -1.0f, 1.0f);
          e.estimated_center_dx_px = est_center_dx_px;
          e.angle_deg = feat->angle_deg;
          e.angle_to_red_signed_deg = feat->angle_to_red_signed_deg;

          single_edges.push_back(e);
        }
      }
    }

    std::sort(center_dets.begin(), center_dets.end(), [](const auto& a, const auto& b) {
      return a.first.angle_to_vertical_abs_err < b.first.angle_to_vertical_abs_err;
    });

    std::optional<LineFeat> lane1_feat;
    std::optional<LineFeat> lane2_feat;
    std::optional<LaneStruct> lane1_struct;
    std::optional<LaneStruct> lane2_struct;

    if (!center_dets.empty()) {
      lane1_feat = center_dets[0].first;
      lane1_struct = center_dets[0].second;
    }
    if (center_dets.size() >= 2) {
      lane2_feat = center_dets[1].first;
      lane2_struct = center_dets[1].second;
    }

    bool intersection = lane1_feat.has_value() && lane2_feat.has_value();

    std::optional<LineFeat> selected_feat;
    std::optional<LaneStruct> selected_struct;
    int selected_lane_id = 0;

    if (lane1_feat.has_value() && !lane2_feat.has_value()) {
      selected_lane_id_ = 1;
    } else if (lane2_feat.has_value() && !lane1_feat.has_value()) {
      selected_lane_id_ = 2;
    }

    if (selected_lane_id_ == 1 && lane1_feat.has_value()) {
      selected_lane_id = 1;
      selected_feat = lane1_feat;
      selected_struct = lane1_struct;
    } else if (selected_lane_id_ == 2 && lane2_feat.has_value()) {
      selected_lane_id = 2;
      selected_feat = lane2_feat;
      selected_struct = lane2_struct;
    } else {
      if (lane1_feat.has_value()) {
        selected_lane_id_ = 1;
        selected_lane_id = 1;
        selected_feat = lane1_feat;
        selected_struct = lane1_struct;
      } else if (lane2_feat.has_value()) {
        selected_lane_id_ = 2;
        selected_lane_id = 2;
        selected_feat = lane2_feat;
        selected_struct = lane2_struct;
      }
    }

    auto draw_center = [&](const std::optional<LaneStruct>& s, const cv::Scalar& color) {
      if (!s.has_value() || !s->center.has_value()) return;
      auto ext = extend_to_borders((*s->center)[0], (*s->center)[1], (*s->center)[2], (*s->center)[3], W_TARGET, H_TARGET);
      if (!ext) return;
      cv::line(vis_, cv::Point((*ext)[0], (*ext)[1]), cv::Point((*ext)[2], (*ext)[3]), color,
               params_.center_thickness, DRAW_LINE_TYPE);
    };

    if (selected_lane_id == 1) {
      draw_center(lane1_struct, COLOR_SELECTED);
      draw_center(lane2_struct, COLOR_OTHER);
    } else if (selected_lane_id == 2) {
      draw_center(lane2_struct, COLOR_SELECTED);
      draw_center(lane1_struct, COLOR_OTHER);
    } else {
      draw_center(lane1_struct, COLOR_OTHER);
      draw_center(lane2_struct, COLOR_OTHER);
    }

    std::optional<SelectedLineCenter> selected_line_center;
    if (selected_struct.has_value() && selected_struct->center.has_value()) {
      auto ext = extend_to_borders((*selected_struct->center)[0], (*selected_struct->center)[1],
                                  (*selected_struct->center)[2], (*selected_struct->center)[3],
                                  W_TARGET, H_TARGET);
      if (ext) {
        float xm = 0.5f * ((*ext)[0] + (*ext)[2]);
        float ym = 0.5f * ((*ext)[1] + (*ext)[3]);
        xm = clampf(xm, 0.0f, static_cast<float>(W_TARGET - 1));
        ym = clampf(ym, 0.0f, static_cast<float>(H_TARGET - 1));
        float dx_px = xm - static_cast<float>(W_TARGET / 2);
        float dx_norm = dx_px / (W_TARGET * 0.5f + 1e-9f);

        cv::circle(vis_, cv::Point(static_cast<int>(std::lround(xm)), static_cast<int>(std::lround(ym))),
                   SELECTED_POINT_RADIUS, SELECTED_POINT_COLOR, -1, DRAW_LINE_TYPE);

        selected_line_center = SelectedLineCenter{xm, ym, dx_px, dx_norm};
      }
    }

    sensor_msgs::msg::Image out_img;
    out_img.header = msg->header;
    out_img.height = static_cast<uint32_t>(vis_.rows);
    out_img.width = static_cast<uint32_t>(vis_.cols);
    out_img.encoding = "bgr8";
    out_img.is_bigendian = false;
    out_img.step = static_cast<uint32_t>(vis_.cols * 3);
    out_img.data.assign(vis_.data, vis_.data + (vis_.rows * vis_.cols * 3));
    pub_img_->publish(out_img);

    float lane1_conf = lane1_struct.has_value() ? compute_confidence(*lane1_struct) : 0.0f;
    float lane2_conf = lane2_struct.has_value() ? compute_confidence(*lane2_struct) : 0.0f;
    float sel_conf = selected_struct.has_value() ? compute_confidence(*selected_struct) : 0.0f;

    std::ostringstream ss;
    ss << std::setprecision(6);

    ss << "{";
    ss << "\"stamp\":{\"sec\":" << static_cast<int>(msg->header.stamp.sec)
       << ",\"nanosec\":" << static_cast<int>(msg->header.stamp.nanosec) << "},";
    ss << "\"img\":{\"w\":" << W_TARGET << ",\"h\":" << H_TARGET
       << ",\"center_x\":" << (W_TARGET / 2) << ",\"bottom_y\":" << (H_TARGET - 1)
       << ",\"mid_y\":" << (H_TARGET / 2) << "},";

    ss << "\"intersection\":" << (intersection ? "true" : "false") << ",";
    ss << "\"lane_count_centerline\":" << static_cast<int>(center_dets.size()) << ",";
    ss << "\"lane_count_single_edge\":" << static_cast<int>(single_edges.size()) << ",";

    ss << "\"lane_select\":{\"selected_lane_id\":";
    if (selected_lane_id_ == 1 || selected_lane_id_ == 2) {
      ss << selected_lane_id_;
    } else {
      ss << "null";
    }
    ss << ",\"selected_visible\":" << (selected_feat.has_value() ? "true" : "false") << "},";

    ss << "\"lane1\":"
       << pack_center_json(lane1_feat, 1, lane1_conf,
                           lane1_struct.has_value() && lane1_struct->edges.size() == 2,
                           lane1_struct.has_value() ? static_cast<int>(lane1_struct->edges.size()) : 0)
       << ",";
    ss << "\"lane2\":"
       << pack_center_json(lane2_feat, 2, lane2_conf,
                           lane2_struct.has_value() && lane2_struct->edges.size() == 2,
                           lane2_struct.has_value() ? static_cast<int>(lane2_struct->edges.size()) : 0)
       << ",";
    ss << "\"selected\":"
       << pack_center_json(selected_feat, selected_lane_id, sel_conf,
                           selected_struct.has_value() && selected_struct->edges.size() == 2,
                           selected_struct.has_value() ? static_cast<int>(selected_struct->edges.size()) : 0)
       << ",";

    ss << "\"selected_line_center\":";
    if (selected_line_center.has_value()) {
      const auto& slc = *selected_line_center;
      ss << "{\"x_px\":" << slc.x_px
         << ",\"y_px\":" << slc.y_px
         << ",\"dx_px\":" << slc.dx_px
         << ",\"dx_norm\":" << slc.dx_norm << "}";
    } else {
      ss << "null";
    }
    ss << ",";

    ss << "\"lanes_centerline\":[";
    for (size_t i = 0; i < center_dets.size(); ++i) {
      const auto& d = center_dets[i];
      ss << (i == 0 ? "" : ",")
         << "{\"confidence\":" << compute_confidence(d.second)
         << ",\"has_pair\":" << (d.second.edges.size() == 2 ? "true" : "false")
         << ",\"edges_count\":" << static_cast<int>(d.second.edges.size())
         << ",\"angle_deg\":" << d.first.angle_deg
         << ",\"angle_to_red_signed_deg\":" << d.first.angle_to_red_signed_deg
         << ",\"bottom_intersect_x_norm\":" << d.first.bottom_intersect_x_norm
         << ",\"mid_intersect_x_norm\":" << d.first.mid_intersect_x_norm
         << ",\"center_offset_x_norm\":" << d.first.center_offset_x_norm
         << ",\"bottom_intersect_dx_px\":" << d.first.bottom_intersect_dx_px
         << ",\"mid_intersect_dx_px\":" << d.first.mid_intersect_dx_px
         << ",\"center_offset_x_px\":" << d.first.center_offset_x_px
         << ",\"angle_to_vertical_abs_err\":" << d.first.angle_to_vertical_abs_err
         << "}";
    }
    ss << "],";

    ss << "\"lanes_single_edge\":[";
    for (size_t i = 0; i < single_edges.size(); ++i) {
      const auto& e = single_edges[i];
      ss << (i == 0 ? "" : ",")
         << "{\"confidence\":" << e.confidence
         << ",\"edge_bottom_dx_norm\":" << e.edge_bottom_dx_norm
         << ",\"edge_bottom_dx_px\":" << e.edge_bottom_dx_px
         << ",\"search_dir\":\"" << json_escape(e.search_dir) << "\""
         << ",\"estimated_center_dx_norm\":" << e.estimated_center_dx_norm
         << ",\"estimated_center_dx_px\":" << e.estimated_center_dx_px
         << ",\"angle_deg\":" << e.angle_deg
         << ",\"angle_to_red_signed_deg\":" << e.angle_to_red_signed_deg
         << "}";
    }
    ss << "]";

    ss << "}";

    std_msgs::msg::String out;
    out.data = ss.str();
    pub_info_->publish(out);

    fps_n_++;
    double now = this->now().seconds();
    if (now - fps_t_ >= 2.0) {
      double fps = fps_n_ / (now - fps_t_);
      RCLCPP_INFO(get_logger(), "lane_centerline_node_cpp fps ~ %.2f", fps);
      fps_t_ = now;
      fps_n_ = 0;
    }
  }

  std::string in_topic_;
  std::string out_topic_;
  std::string info_topic_;
  std::string change_lane_topic_;
  int skip_n_ = 0;
  float lane_half_width_px_ = 60.0f;
  Params params_;

  cv::Mat vis_;
  cv::Mat cache_k_grad_;
  cv::Mat cache_k_thick_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_info_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_change_;

  int selected_lane_id_ = 1;
  int frame_i_ = 0;
  double fps_t_ = 0.0;
  int fps_n_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LaneCenterlineNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
