#!/usr/bin/env python3
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, UInt16, Int16, Empty as EmptyMsg

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class DemoCoreNode(Node):
    def __init__(self):
        super().__init__("demo_core")

        # -------- Parameters --------
        self.declare_parameter("sounds_dir", str(Path.home() / "Sounds"))
        self.declare_parameter("audio_topic", "/audio/path")
        self.declare_parameter("servo_topic", "/servo/angle_deg")
        self.declare_parameter("analyze_offset_deg", 40.0)
        self.declare_parameter("analyze_delay_sec", 0.6)
        self.declare_parameter("analyze_sound", "analyze.mp3")
        self.declare_parameter("slogan_sound", "slogan.mp3")
        self.declare_parameter("tts_enable", True)
        self.declare_parameter("tts_model", "gpt-4o-mini-tts")
        self.declare_parameter("tts_voice", "ballad")
        self.declare_parameter("tts_style", "Bir robotun içinden konuşuyormuş gibi hafif mekanik, enerjik ve sevimli bir ton kullan.")
        self.declare_parameter("text_model", "gpt-4o-mini")
        self.declare_parameter("lcd_media_topic", "/lcd/media_path")
        self.declare_parameter("lcd_stop_topic", "/lcd/stop")
        self.declare_parameter("measure_video_dir", str(Path.home() / "Emojis"))
        self.declare_parameter("measure_air_video", "hava_kalitesi_olculuyor.mp4")
        self.declare_parameter("measure_env_video", "sicaklik_nem_olculuyor.mp4")
        self.declare_parameter("measure_sound_video", "ses_seviyesi_olculuyor.mp4")

        self.sounds_dir = Path(self.get_parameter("sounds_dir").value)
        self.audio_topic = str(self.get_parameter("audio_topic").value)
        self.servo_topic = str(self.get_parameter("servo_topic").value)
        self.analyze_offset = float(self.get_parameter("analyze_offset_deg").value)
        self.analyze_delay = float(self.get_parameter("analyze_delay_sec").value)
        self.analyze_sound = str(self.get_parameter("analyze_sound").value)
        self.slogan_sound = str(self.get_parameter("slogan_sound").value)
        self.tts_enable = bool(self.get_parameter("tts_enable").value)
        self.tts_model = str(self.get_parameter("tts_model").value)
        self.tts_voice = str(self.get_parameter("tts_voice").value)
        self.tts_style = str(self.get_parameter("tts_style").value)
        self.text_model = str(self.get_parameter("text_model").value)
        self.lcd_media_topic = str(self.get_parameter("lcd_media_topic").value)
        self.lcd_stop_topic = str(self.get_parameter("lcd_stop_topic").value)
        self.measure_video_dir = Path(self.get_parameter("measure_video_dir").value)
        self.measure_air_video = str(self.get_parameter("measure_air_video").value)
        self.measure_env_video = str(self.get_parameter("measure_env_video").value)
        self.measure_sound_video = str(self.get_parameter("measure_sound_video").value)

        self.sounds_dir.mkdir(parents=True, exist_ok=True)

        # -------- Publishers --------
        self.pub_audio = self.create_publisher(String, self.audio_topic, 10)
        self.pub_servo = self.create_publisher(Float32, self.servo_topic, 10)
        self.pub_lcd_media = self.create_publisher(String, self.lcd_media_topic, 10)
        self.pub_lcd_stop = self.create_publisher(EmptyMsg, self.lcd_stop_topic, 10)

        # -------- Subscriptions --------
        self.segment_topics = {
            "battery": "/segment/battery",
            "plastic": "/segment/plastic",
            "metal": "/segment/metal",
            "glass": "/segment/glass",
            "paper": "/segment/paper",
        }

        for segment, topic in self.segment_topics.items():
            self.create_subscription(String, topic, lambda msg, s=segment: self._on_segment(msg, s), 10)

        self.create_subscription(String, "/robot/slogan", self._on_slogan, 10)
        self.create_subscription(String, "/robot/analyze", self._on_analyze, 10)

        # -------- Measure commands (mobile) --------
        self.create_subscription(String, "/air_quality", self._on_air_quality_measure, 10)
        self.create_subscription(String, "/environment", self._on_environment_measure, 10)
        self.create_subscription(String, "/audio/measure", self._on_sound_measure, 10)

        # -------- Sensor telemetry (pico_com) --------
        self.last_eco2_ppm = None
        self.last_tvoc_ppb = None
        self.last_temp_c = None
        self.last_hum_pct = None
        self.last_rms = None
        self.last_peak = None

        self.create_subscription(UInt16, "/pico/ccs811/eco2_ppm", self._on_eco2, 10)
        self.create_subscription(UInt16, "/pico/ccs811/tvoc_ppb", self._on_tvoc, 10)
        self.create_subscription(Int16, "/pico/dht22/temp_c_x100", self._on_temp, 10)
        self.create_subscription(UInt16, "/pico/dht22/hum_pct_x100", self._on_hum, 10)
        self.create_subscription(UInt16, "/pico/mic/rms", self._on_rms, 10)
        self.create_subscription(UInt16, "/pico/mic/peak", self._on_peak, 10)

        # -------- Worker --------
        self._queue: Queue[Tuple[str, Optional[str]]] = Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"Demo core ready | sounds_dir={self.sounds_dir} audio_topic={self.audio_topic} "
            f"servo_topic={self.servo_topic} tts_enable={self.tts_enable}"
        )

        try:
            rclpy.get_default_context().on_shutdown(self._on_shutdown)
        except Exception:
            pass

    # ---------- Callbacks ----------
    def _on_segment(self, msg: String, segment: str):
        if not self._is_trigger(msg.data):
            return
        self._queue.put(("segment", segment))

    def _on_slogan(self, msg: String):
        if not self._is_trigger(msg.data):
            return
        self._queue.put(("slogan", None))

    def _on_analyze(self, msg: String):
        if not self._is_trigger(msg.data):
            return
        self._queue.put(("analyze", None))

    def _on_air_quality_measure(self, msg: String):
        if not self._is_measure_cmd(msg.data):
            return
        self._queue.put(("measure_air", None))

    def _on_environment_measure(self, msg: String):
        if not self._is_measure_cmd(msg.data):
            return
        self._queue.put(("measure_env", None))

    def _on_sound_measure(self, msg: String):
        if not self._is_measure_cmd(msg.data):
            return
        self._queue.put(("measure_sound", None))

    @staticmethod
    def _is_trigger(value: str) -> bool:
        v = (value or "").strip().lower()
        if v in ("1", "true", "on", "yes"):
            return True
        try:
            return int(v) == 1
        except Exception:
            return False

    @staticmethod
    def _is_measure_cmd(value: str) -> bool:
        v = (value or "").strip().lower()
        return v in ("measure", "start", "go")

    # ---------- Worker ----------
    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                action, value = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if action == "segment" and value:
                    self._play_segment_sound(value)
                elif action == "slogan":
                    self._play_specific_sound(self.slogan_sound)
                elif action == "analyze":
                    self._play_analyze()
                elif action == "measure_air":
                    self._measure_air_quality()
                elif action == "measure_env":
                    self._measure_environment()
                elif action == "measure_sound":
                    self._measure_sound()
            except Exception as exc:
                self.get_logger().error(f"Action failed: {exc}")

    # ---------- Audio helpers ----------
    def _publish_audio_path(self, path: Path):
        msg = String()
        msg.data = str(path)
        self.pub_audio.publish(msg)

    def _publish_lcd_media(self, path: Path):
        msg = String()
        msg.data = str(path)
        self.pub_lcd_media.publish(msg)

    def _publish_lcd_stop(self):
        self.pub_lcd_stop.publish(EmptyMsg())

    def _find_prefix_sound(self, prefix: str) -> Optional[Path]:
        if not self.sounds_dir.exists():
            return None

        candidates = [
            p for p in self.sounds_dir.iterdir()
            if p.is_file() and p.name.startswith(prefix)
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    def _play_segment_sound(self, segment: str):
        prefix = f"segment-{segment}"
        selected = self._find_prefix_sound(prefix)
        if not selected:
            self.get_logger().warn(f"No sound found for segment '{segment}' with prefix '{prefix}'")
            return
        self.get_logger().info(f"Play segment sound: {selected.name}")
        self._publish_audio_path(selected)

    def _play_specific_sound(self, filename: str):
        path = self.sounds_dir / filename
        if not path.exists():
            self.get_logger().warn(f"Sound not found: {path}")
            return
        self.get_logger().info(f"Play sound: {path.name}")
        self._publish_audio_path(path)

    def _play_analyze(self):
        # Play analyze sound
        if not self._try_play_analyze_sound():
            self.get_logger().warn("Analyze sound not found (analyze.mp3/analyse.mp3)")

        # Servo +offset then -offset
        self._publish_servo(self.analyze_offset)
        time.sleep(max(0.0, self.analyze_delay))
        self._publish_servo(-self.analyze_offset)

    def _try_play_analyze_sound(self) -> bool:
        # Try configured name first, then fallback to common spellings
        for name in [self.analyze_sound, "analyze.mp3", "analyse.mp3"]:
            path = self.sounds_dir / name
            if path.exists():
                self.get_logger().info(f"Play analyze sound: {path.name}")
                self._publish_audio_path(path)
                return True
        return False

    # ---------- Servo helpers ----------
    def _publish_servo(self, offset_deg: float):
        msg = Float32()
        msg.data = float(offset_deg)
        self.pub_servo.publish(msg)

    # ---------- Pico telemetry callbacks ----------
    def _on_eco2(self, msg: UInt16):
        self.last_eco2_ppm = int(msg.data)

    def _on_tvoc(self, msg: UInt16):
        self.last_tvoc_ppb = int(msg.data)

    def _on_temp(self, msg: Int16):
        self.last_temp_c = int(msg.data) / 100.0

    def _on_hum(self, msg: UInt16):
        self.last_hum_pct = int(msg.data) / 100.0

    def _on_rms(self, msg: UInt16):
        self.last_rms = int(msg.data)

    def _on_peak(self, msg: UInt16):
        self.last_peak = int(msg.data)

    # ---------- Measurement actions ----------
    def _measure_air_quality(self):
        self._play_measure_video(self.measure_air_video)
        text = self._generate_ai_text(
            kind="air_quality",
            fallback=self._compose_air_quality_text(),
            data=self._air_quality_payload(),
        )
        self._speak_text(text, tag="air")
        self._publish_lcd_stop()

    def _measure_environment(self):
        self._play_measure_video(self.measure_env_video)
        text = self._generate_ai_text(
            kind="environment",
            fallback=self._compose_environment_text(),
            data=self._environment_payload(),
        )
        self._speak_text(text, tag="env")
        self._publish_lcd_stop()

    def _measure_sound(self):
        self._play_measure_video(self.measure_sound_video)
        text = self._generate_ai_text(
            kind="sound",
            fallback=self._compose_sound_text(),
            data=self._sound_payload(),
        )
        self._speak_text(text, tag="sound")
        self._publish_lcd_stop()

    def _play_measure_video(self, filename: str):
        path = self.measure_video_dir / filename
        if not path.exists():
            self.get_logger().warn(f"Measure video not found: {path}")
            return
        self._publish_lcd_media(path)

    # ---------- Message generation ----------
    def _compose_air_quality_text(self) -> str:
        eco2 = self.last_eco2_ppm
        tvoc = self.last_tvoc_ppb
        if eco2 is None and tvoc is None:
            return "Hava kalitesi ölçümü için veri gelmedi. Lütfen sensörü kontrol eder misin?"

        status = "iyi"
        if eco2 is not None and eco2 >= 1000:
            status = "kötü"
        elif eco2 is not None and eco2 >= 700:
            status = "orta"

        parts = ["Hava kalitesi ölçümü tamamlandı."]
        if status == "iyi":
            parts.append(random.choice([
                "Hava harika ve ferah görünüyor.",
                "Hava tertemiz, nefes almak çok keyifli.",
                "Hava çok güzel, içeriği ferah.",
            ]))
        elif status == "orta":
            parts.append(random.choice([
                "Hava kalitesi orta seviyede, biraz havalandırma iyi olur.",
                "Hava idare eder, pencereyi kısa süre açabiliriz.",
                "Hava fena değil ama biraz tazelemek iyi gelir.",
            ]))
        else:
            parts.append(random.choice([
                "Hava biraz kirli, lütfen ortamı havalandıralım.",
                "Hava tazelenmeye ihtiyaç duyuyor, pencere açalım.",
                "Hava ağırlaşmış, biraz hava değişimi iyi olur.",
            ]))

        return " ".join(parts)

    def _air_quality_payload(self) -> dict:
        return {
            "eco2_ppm": self.last_eco2_ppm,
            "tvoc_ppb": self.last_tvoc_ppb,
        }

    def _compose_environment_text(self) -> str:
        temp = self.last_temp_c
        hum = self.last_hum_pct
        if temp is None and hum is None:
            return "Ortam ölçümü için veri gelmedi. Lütfen sensörü kontrol eder misin?"

        parts = ["Ortam ölçümü tamamlandı."]
        if temp is not None:
            parts.append(f"Sıcaklık {temp:.1f} santigrat derece.")
        if hum is not None:
            parts.append(f"Nem yüzde {hum:.1f}.")

        if temp is not None and temp >= 30:
            parts.append("Biraz sıcak. Serin bir ortama geçebilirsin.")
        elif temp is not None and temp <= 16:
            parts.append("Hava serin. Üşümemek için dikkatli ol.")

        if hum is not None and hum >= 70:
            parts.append("Nem yüksek, ortamı havalandırmak iyi olabilir.")
        elif hum is not None and hum <= 30:
            parts.append("Nem düşük, cilt kuruluğu olabilir. Su tüketimini artır.")

        return " ".join(parts)

    def _environment_payload(self) -> dict:
        return {
            "temp_c": self.last_temp_c,
            "humidity_pct": self.last_hum_pct,
        }

    def _compose_sound_text(self) -> str:
        rms = self.last_rms
        peak = self.last_peak
        if rms is None and peak is None:
            return "Ses ölçümü için veri gelmedi. Lütfen mikrofonu kontrol eder misin?"

        noise_pct = None
        if rms is not None:
            noise_pct = min(100, max(0, int((rms / 1023.0) * 100)))

        parts = ["Ses seviyesi ölçümü tamamlandı."]
        if noise_pct is not None:
            if noise_pct >= 80:
                parts.append(random.choice([
                    "Ortam çok gürültülü. Daha sessiz bir yere geçebilirsin.",
                    "Burası epey gürültülü, biraz sessiz bir alan iyi olur.",
                    "Gürültü yüksek, mümkünse daha sakin bir yere geçelim.",
                ]))
            elif noise_pct >= 55:
                parts.append(random.choice([
                    "Ortam biraz sesli, konuşmak zorlaşabilir.",
                    "Ortam orta derecede gürültülü.",
                    "Biraz ses var, dikkat dağılabilir.",
                ]))
            else:
                parts.append(random.choice([
                    "Şu an ortam sakin ve sessiz görünüyor.",
                    "Burası oldukça sessiz ve rahat.",
                    "Ses seviyesi düşük, ortam huzurlu.",
                ]))
        else:
            parts.append("Ses durumunu anlayamadım. Mikrofonu kontrol edebilir misin?")

        return " ".join(parts)

    def _sound_payload(self) -> dict:
        rms = self.last_rms
        peak = self.last_peak
        noise_pct = None
        if rms is not None:
            noise_pct = min(100, max(0, int((rms / 1023.0) * 100)))
        return {
            "rms": rms,
            "peak": peak,
            "noise_pct_est": noise_pct,
        }

    def _generate_ai_text(self, kind: str, fallback: str, data: dict) -> str:
        if not self.tts_enable or OpenAI is None:
            return fallback

        try:
            client = OpenAI()
            seed = random.randint(1000, 9999)
            prompt = (
                "Aşağıdaki sensör verilerine göre kısa, net ve kullanıcı dostu bir Türkçe bildirim üret. "
                "Robotun sevimli ve yardımsever tonu olsun. 1-3 cümle yeterli. "
                "Hava kalitesi ve ses için sayısal değerleri veya birimleri söyleme. "
                "Sadece niteliksel yorum yap (harika, iyi, orta, gürültülü gibi). "
                "Sıcaklık ve nem için sayısal değer söyleyebilirsin. "
                "Aynı ifadeyi tekrar etme; farklı bir anlatım kullan. "
                "Veri yoksa bunu belirt ve sensörü kontrol etmesini iste. "
                "Asla soru sorma. Asla 'başka bir konuda yardımcı olayım mı' gibi cümleler kurma. "
                "Model, asistan, yapay zeka gibi ifadeler kullanma.\n"
                f"Konu: {kind}\nVeri: {data}\nVaryasyon: {seed}"
            )
            resp = client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "Sen sevimli ve yardımsever bir robotsun. Türkçe, net ve kısa konuş."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = self._sanitize_ai_text(text)
            if not text:
                return fallback
            return text
        except Exception as exc:
            self.get_logger().warn(f"AI text generation failed: {exc}")
            return fallback

    # ---------- TTS ----------
    def _speak_text(self, text: str, tag: str):
        if not self.tts_enable:
            self.get_logger().warn("TTS disabled (tts_enable=false)")
            return
        if OpenAI is None:
            self.get_logger().error("OpenAI SDK not available. Install openai package.")
            return

        client = OpenAI()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.sounds_dir / f"tts_{tag}_{ts}.mp3"

        try:
            response = client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
            )
            response.stream_to_file(out_path)

            self.get_logger().info(f"TTS generated: {out_path} | text='{text}'")
            self._publish_audio_path(out_path)
        except Exception as exc:
            self.get_logger().error(f"TTS failed: {exc}")

    # ---------- Shutdown ----------
    def _on_shutdown(self):
        self._stop_event.set()

    @staticmethod
    def _sanitize_ai_text(text: str) -> str:
        if not text:
            return text
        banned_phrases = [
            "başka bir konuda yardımcı", "başka konuda yardımcı", "yardımcı olabilir miyim",
            "yardımcı olayım", "başka bir şey", "sana başka", "yardım edeyim",
            "soru sormak", "başka bir sor", "yardım ister misin",
            "model", "asistan", "yapay zeka",
        ]
        # Split into sentences and drop any that contain banned phrases
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
        kept = []
        for s in sentences:
            if not s:
                continue
            low = s.lower()
            if any(p in low for p in banned_phrases):
                continue
            kept.append(s)
        return ". ".join(kept).strip()


def main(args=None):
    rclpy.init(args=args)
    node = DemoCoreNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
