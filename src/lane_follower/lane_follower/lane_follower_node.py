#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, String

from cv_bridge import CvBridge
import cv2
import numpy as np
import time


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__("lane_follower_node")

        # ---------- Params ----------
        self.declare_parameter("image_topic", "/camera_bottom/camera_node/image_raw")
        self.declare_parameter("motor_topic", "/motor_cmd")
        self.declare_parameter("debug_image_topic", "/lane_follower/debug_image")
        self.declare_parameter("publish_debug_image", True)

        self.declare_parameter("base_speed", 350)      # ileri hız (0..1000)
        self.declare_parameter("kp", 700.0)            # P kazanç (deneyerek)
        self.declare_parameter("max_speed", 800)       # saturasyon
        self.declare_parameter("max_turn", 400)        # direksiyon limiti
        self.declare_parameter("err_deadband_px", 12)  # merkezdeki tolerans
        self.declare_parameter("turn_smoothing", 0.3)  # 0.0=kapali, 0.9=yavas
        self.declare_parameter("invert_steering", False)

        # ROI: alt bölgeden şerit bulacağız
        self.declare_parameter("roi_height_ratio", 0.35)  # görüntünün alt %35'i

        # Beyaz şerit algılama
        self.declare_parameter("blur_ksize", 5)
        self.declare_parameter("clahe_clip", 3.0)       # CLAHE kontrast
        self.declare_parameter("adaptive_block", 51)     # adaptif esik blok (tek sayi)
        self.declare_parameter("adaptive_c", -15)        # adaptif esik C (negatif=parlak sec)
        self.declare_parameter("sat_max", 60)            # HSV saturasyon ust limiti (beyaz=dusuk S)
        self.declare_parameter("val_min", 100)           # HSV value alt limiti
        self.declare_parameter("min_contour_area", 200)
        self.declare_parameter("min_aspect_ratio", 0.8)

        # Morfoloji
        self.declare_parameter("morph_kernel", 5)

        # --- Kesişim algılama ---
        self.declare_parameter("intersection_white_ratio", 0.30)  # ROI'nin %30'u beyazsa kesişim
        self.declare_parameter("intersection_width_ratio", 0.70)  # beyaz alan ROI genisliginin %70'ini kapliyorsa
        self.declare_parameter("intersection_confirm_frames", 5)  # art arda N frame onay
        self.declare_parameter("turn_cmd_topic", "/lane_follower/turn_cmd")

        # --- Dönüş manevrası ---
        self.declare_parameter("turn_speed", 350)        # dönüş hızı
        self.declare_parameter("turn_duration", 1.2)     # dönüş süresi (saniye)
        self.declare_parameter("forward_after_turn", 0.5) # dönüşten sonra düz git (saniye)

        # Debug (istersen pencere açar - headless ise kapalı tut)
        self.declare_parameter("debug_view", False)

        # ---------- IO ----------
        self.bridge = CvBridge()

        img_topic = self.get_parameter("image_topic").value
        motor_topic = self.get_parameter("motor_topic").value
        debug_img_topic = self.get_parameter("debug_image_topic").value

        turn_cmd_topic = self.get_parameter("turn_cmd_topic").value

        self.pub = self.create_publisher(Int16MultiArray, motor_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_img_topic, 10)
        self.status_pub = self.create_publisher(String, "/lane_follower/status", 10)
        self.sub = self.create_subscription(Image, img_topic, self.on_image, 10)
        self.turn_sub = self.create_subscription(String, turn_cmd_topic, self.on_turn_cmd, 10)

        self.last_motor_sub_warn = self.get_clock().now()

        # --- Durum makinesi ---
        # FOLLOWING  : normal şerit takibi
        # WAITING    : kesişimde durdu, komut bekliyor
        # TURNING    : dönüş manevrası yapıyor
        self.state = "FOLLOWING"
        self.intersection_counter = 0
        self.pending_turn = None       # "left" veya "right"
        self.turn_start_time = None
        self.turn_phase = None         # "rotate" veya "forward"

        self.get_logger().info(
            f"lane_follower READY | sub={img_topic} pub={motor_topic} turn_cmd={turn_cmd_topic}"
        )
        self.get_logger().info("Kesişimde durur, /lane_follower/turn_cmd 'left' veya 'right' bekler")

        self.prev_turn = 0

    # ========== Dönüş komutu callback ==========
    def on_turn_cmd(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd not in ("left", "right"):
            self.get_logger().warn(f"Geçersiz turn komutu: '{cmd}' — 'left' veya 'right' gönder")
            return

        if self.state == "WAITING":
            self.pending_turn = cmd
            self.state = "TURNING"
            self.turn_phase = "rotate"
            self.turn_start_time = time.time()
            self.get_logger().info(f"Dönüş başladı: {cmd}")
        else:
            self.get_logger().info(f"Turn komutu alındı ({cmd}) ama state={self.state}, WAITING değil")

    # ========== Ana görüntü callback ==========
    def on_image(self, msg: Image):
        # --- TURNING state ise manevraya devam et ---
        if self.state == "TURNING":
            self.execute_turn(msg)
            return

        # Params her frame okunabilir (kolay tuning)
        base_speed = int(self.get_parameter("base_speed").value)
        kp = float(self.get_parameter("kp").value)
        max_speed = int(self.get_parameter("max_speed").value)
        max_turn = int(self.get_parameter("max_turn").value)
        err_deadband_px = int(self.get_parameter("err_deadband_px").value)
        turn_smoothing = float(self.get_parameter("turn_smoothing").value)
        invert_steering = bool(self.get_parameter("invert_steering").value)
        roi_h_ratio = float(self.get_parameter("roi_height_ratio").value)

        blur_ksize = int(self.get_parameter("blur_ksize").value)
        clahe_clip = float(self.get_parameter("clahe_clip").value)
        adaptive_block = int(self.get_parameter("adaptive_block").value)
        adaptive_c = int(self.get_parameter("adaptive_c").value)
        sat_max = int(self.get_parameter("sat_max").value)
        val_min = int(self.get_parameter("val_min").value)
        min_contour_area = int(self.get_parameter("min_contour_area").value)
        min_aspect_ratio = float(self.get_parameter("min_aspect_ratio").value)

        intersection_white_ratio = float(self.get_parameter("intersection_white_ratio").value)
        intersection_width_ratio = float(self.get_parameter("intersection_width_ratio").value)
        intersection_confirm = int(self.get_parameter("intersection_confirm_frames").value)

        k = int(self.get_parameter("morph_kernel").value)
        debug = bool(self.get_parameter("debug_view").value)
        publish_debug_image = bool(self.get_parameter("publish_debug_image").value)

        # --- ROS Image -> OpenCV ---
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge convert failed: {e}")
            return

        # --- 180 derece döndür (ters kamera) ---
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]
        roi_h = int(h * roi_h_ratio)
        y0 = h - roi_h
        roi = frame[y0:h, 0:w]

        # --- Mask oluştur (çoklu yöntem OR birleşimi) ---
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blur_ksize = max(1, blur_ksize)

        # 1) Gri kanal + CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # 2) Otsu: otomatik en iyi eşiği bulur
        _, mask_otsu = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3) Adaptif eşikleme: lokal parlak bölgeleri yakala
        adaptive_block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        adaptive_block = max(3, adaptive_block)
        mask_adapt = cv2.adaptiveThreshold(
            gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_block, adaptive_c
        )

        # 4) LAB L kanalı: aydınlıktan bağımsız beyaz algılama
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        l_mean = np.mean(l_channel)
        l_thresh = max(int(l_mean + 20), 140)
        _, mask_lab = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY)

        # 5) Üçünü OR ile birleştir (en az birinde beyaz olan = beyaz)
        mask_combined = cv2.bitwise_or(mask_otsu, mask_adapt)
        mask_combined = cv2.bitwise_or(mask_combined, mask_lab)

        # 6) HSV saturasyon filtresi ile son temizlik (AND)
        #    beyaz = düşük S, yüksek V — sadece gerçek beyazları geçir
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_max_clamped = min(sat_max, 255)
        val_min_clamped = max(val_min, 0)
        mask_white = cv2.inRange(
            hsv,
            np.array([0, 0, val_min_clamped], dtype=np.uint8),
            np.array([179, sat_max_clamped, 255], dtype=np.uint8)
        )
        mask = cv2.bitwise_and(mask_combined, mask_white)

        # --- Gürültü temizle ---
        k = max(1, k)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- Konturları filtrele: şerit olma ihtimali düşük olanları at ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box == 0:
                continue
            aspect = h_box / float(w_box)
            if aspect < min_aspect_ratio:
                continue
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

        mask = filtered

        # ====== KESİŞİM ALGILAMA ======
        roi_total_px = mask.shape[0] * mask.shape[1]
        white_px = cv2.countNonZero(mask)
        white_ratio = white_px / float(roi_total_px) if roi_total_px > 0 else 0.0

        # Beyaz alanın yatay yayılımını ölç
        col_sum = np.sum(mask > 0, axis=0)  # her sütundaki beyaz piksel sayısı
        active_cols = np.sum(col_sum > (mask.shape[0] * 0.05))  # en az %5 dolu sütunlar
        width_ratio = active_cols / float(w) if w > 0 else 0.0

        is_intersection = (
            white_ratio > intersection_white_ratio
            and width_ratio > intersection_width_ratio
        )

        if is_intersection and self.state == "FOLLOWING":
            self.intersection_counter += 1
        else:
            self.intersection_counter = 0

        if self.intersection_counter >= intersection_confirm and self.state == "FOLLOWING":
            self.state = "WAITING"
            self.publish_motor(0, 0)
            self.get_logger().info(
                f"KESİŞİM TESPİT EDİLDİ (white={white_ratio:.1%} width={width_ratio:.1%}) — "
                "DUR! Komut bekliyor: ros2 topic pub --once /lane_follower/turn_cmd std_msgs/String 'data: left' veya 'right'"
            )
            status_msg = String()
            status_msg.data = "WAITING_AT_INTERSECTION"
            self.status_pub.publish(status_msg)

        # --- WAITING state: dur ve bekle ---
        if self.state == "WAITING":
            self.publish_motor(0, 0)
            self.publish_debug_image(frame, roi, mask, None, None, 0, 0, publish_debug_image,
                                     state_text="KESISIM - KOMUT BEKLIYOR")
            return

        # ====== NORMAL ŞERİT TAKİBİ ======
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.publish_motor(0, 0)
            self.publish_debug_image(frame, roi, mask, None, None, 0, 0, publish_debug_image,
                                     state_text="SERIT YOK")
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_contour_area:
            self.publish_motor(0, 0)
            self.publish_debug_image(frame, roi, mask, None, None, 0, 0, publish_debug_image)
            return

        M = cv2.moments(largest)
        if M["m00"] == 0:
            self.publish_motor(0, 0)
            self.publish_debug_image(frame, roi, mask, None, None, 0, 0, publish_debug_image)
            return

        cx = int(M["m10"] / M["m00"])
        target = w // 2
        err_px = cx - target

        if abs(err_px) <= err_deadband_px:
            err_px = 0

        err = err_px / float(w // 2)

        turn_raw = int(kp * err)
        if invert_steering:
            turn_raw = -turn_raw
        max_turn = max(0, max_turn)
        if max_turn > 0:
            turn_raw = clamp(turn_raw, -max_turn, max_turn)

        turn_smoothing = clamp(turn_smoothing, 0.0, 0.95)
        turn = int((1.0 - turn_smoothing) * turn_raw + turn_smoothing * self.prev_turn)
        self.prev_turn = turn

        left = base_speed - turn
        right = base_speed + turn

        left = clamp(left, -max_speed, max_speed)
        right = clamp(right, -max_speed, max_speed)

        self.publish_motor(left, right)

        self.publish_debug_image(frame, roi, mask, (cx, roi_h // 2), target, err_px, turn, publish_debug_image,
                                 state_text=f"FOLLOWING w={white_ratio:.0%}")

    # ========== Dönüş manevrası ==========
    def execute_turn(self, msg: Image):
        turn_speed = int(self.get_parameter("turn_speed").value)
        turn_duration = float(self.get_parameter("turn_duration").value)
        forward_after = float(self.get_parameter("forward_after_turn").value)
        base_speed = int(self.get_parameter("base_speed").value)
        publish_debug_image = bool(self.get_parameter("publish_debug_image").value)

        elapsed = time.time() - self.turn_start_time

        # Debug frame
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        except Exception:
            frame = None

        if self.turn_phase == "rotate":
            if elapsed < turn_duration:
                if self.pending_turn == "left":
                    self.publish_motor(-turn_speed, turn_speed)
                else:
                    self.publish_motor(turn_speed, -turn_speed)
                if frame is not None:
                    h, w = frame.shape[:2]
                    roi = frame[int(h*0.6):h, :]
                    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                    self.publish_debug_image(frame, roi, mask, None, None, 0, 0,
                                             publish_debug_image,
                                             state_text=f"DONUS: {self.pending_turn} ({elapsed:.1f}s)")
            else:
                self.turn_phase = "forward"
                self.turn_start_time = time.time()

        elif self.turn_phase == "forward":
            if elapsed < forward_after:
                self.publish_motor(base_speed, base_speed)
                if frame is not None:
                    h, w = frame.shape[:2]
                    roi = frame[int(h*0.6):h, :]
                    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                    self.publish_debug_image(frame, roi, mask, None, None, 0, 0,
                                             publish_debug_image,
                                             state_text=f"DONUS SONRASI ILERI ({elapsed:.1f}s)")
            else:
                self.publish_motor(0, 0)
                self.state = "FOLLOWING"
                self.intersection_counter = 0
                self.pending_turn = None
                self.turn_phase = None
                self.prev_turn = 0
                self.get_logger().info("Dönüş tamamlandı, şerit takibine devam")
                status_msg = String()
                status_msg.data = "FOLLOWING"
                self.status_pub.publish(status_msg)

    # ========== Motor publish ==========
    def publish_motor(self, left: int, right: int):
        if self.pub.get_subscription_count() == 0:
            now = self.get_clock().now()
            if (now - self.last_motor_sub_warn).nanoseconds > 2_000_000_000:
                self.get_logger().warn("motor_topic has no subscribers; motor commands are not received")
                self.last_motor_sub_warn = now
        msg = Int16MultiArray()
        msg.data = [int(left), int(right)]
        self.pub.publish(msg)

    # ========== Debug görüntü publish ==========
    def publish_debug_image(
        self,
        frame,
        roi,
        mask,
        center,
        target,
        err_px,
        turn,
        publish_debug_image,
        state_text="",
    ):
        if not publish_debug_image:
            return

        h, w = frame.shape[:2]
        roi_h = roi.shape[0]
        y0 = h - roi_h

        dbg = frame.copy()
        # ROI sınırını çiz
        cv2.line(dbg, (0, y0), (w, y0), (255, 255, 0), 1)

        if target is not None:
            cv2.line(dbg, (target, y0), (target, h), (0, 255, 0), 2)
        if center is not None:
            cx, cy = center
            cv2.circle(dbg, (cx, y0 + cy), 6, (0, 0, 255), -1)

        # Durum ve bilgi yazısı
        info_y = 25
        if state_text:
            cv2.putText(dbg, state_text, (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        if center is not None:
            cv2.putText(dbg, f"err={err_px} turn={turn}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mask inset (sol üst köşe)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        inset_h = min(roi_h, 150)
        if mask.shape[0] > 0:
            inset_w = int(inset_h * (mask.shape[1] / mask.shape[0]))
            inset_w = max(1, inset_w)
            inset = cv2.resize(mask_bgr, (inset_w, inset_h))
            x_off = w - inset_w
            dbg[0:inset_h, x_off:w] = inset

        try:
            img_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            self.debug_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"debug image publish failed: {e}")


def main():
    rclpy.init()
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()