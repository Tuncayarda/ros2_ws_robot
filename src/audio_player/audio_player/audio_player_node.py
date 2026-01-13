#!/usr/bin/env python3
import os
import signal
import subprocess
import json
import socket
import time
import atexit
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty, Int32


class AudioPlayerNode(Node):
    def __init__(self):
        super().__init__("audio_player_node")

        # -------- Parameters --------
        self.declare_parameter("topic_path", "/audio/path")
        self.declare_parameter("topic_url",  "/audio/url")
        self.declare_parameter("topic_stop", "/audio/stop")
        self.declare_parameter("topic_volume", "/audio/volume")
        self.declare_parameter("topic_volume_state", "/audio/volume_state")
        self.declare_parameter("topic_volume_get", "/audio/volume/get")

        self.declare_parameter("player", "mpv")
        self.declare_parameter("volume", 30)          # 0..100
        self.declare_parameter("ytdl", True)
        self.declare_parameter("alsa_device", "")
        self.declare_parameter("extra_args", [])
        self.declare_parameter("ipc_socket", "/tmp/mpv_audio.sock")

        self.topic_path   = self.get_parameter("topic_path").value
        self.topic_url    = self.get_parameter("topic_url").value
        self.topic_stop   = self.get_parameter("topic_stop").value
        self.topic_volume = self.get_parameter("topic_volume").value
        self.topic_volume_state = self.get_parameter("topic_volume_state").value
        self.topic_volume_get = self.get_parameter("topic_volume_get").value

        self.player = str(self.get_parameter("player").value)
        self.volume = int(self.get_parameter("volume").value)
        self.ytdl   = bool(self.get_parameter("ytdl").value)
        self.alsa_device = str(self.get_parameter("alsa_device").value)
        self.extra_args: List[str] = list(self.get_parameter("extra_args").value)
        self.ipc_socket = str(self.get_parameter("ipc_socket").value)

        self.volume = self._clamp_volume(self.volume)

        self.proc: Optional[subprocess.Popen] = None

        # -------- Publishers/Subscriptions --------
        self.sub_path = self.create_subscription(String, self.topic_path, self.on_path, 10)
        self.sub_url  = self.create_subscription(String, self.topic_url,  self.on_url,  10)
        self.sub_stop = self.create_subscription(Empty,  self.topic_stop, self.on_stop, 10)
        self.sub_vol  = self.create_subscription(Int32,  self.topic_volume, self.on_volume, 10)
        self.sub_vol_get = self.create_subscription(String, self.topic_volume_get, self.on_volume_get, 10)

        self.pub_vol_state = self.create_publisher(Int32, self.topic_volume_state, 10)

        self.get_logger().info(
            f"AudioPlayer ready | path={self.topic_path} url={self.topic_url} stop={self.topic_stop} "
            f"vol_in={self.topic_volume} vol_state={self.topic_volume_state} vol_get={self.topic_volume_get} | "
            f"player={self.player} volume={self.volume} ytdl={int(self.ytdl)} "
            f"alsa_device='{self.alsa_device}' ipc='{self.ipc_socket}'"
        )

        # publish initial volume (subscribed varsa alır)
        self._publish_volume_state(self.volume)

        # ROS kapanırken de stop dene
        try:
            rclpy.get_default_context().on_shutdown(self._on_ros_shutdown)
        except Exception:
            pass

    # ---------- Helpers ----------
    @staticmethod
    def _clamp_volume(v: int) -> int:
        return max(0, min(100, int(v)))

    def _publish_volume_state(self, v: int):
        msg = Int32()
        msg.data = self._clamp_volume(v)
        self.pub_vol_state.publish(msg)

    def _mpv_alive(self) -> bool:
        return self.proc is not None and (self.proc.poll() is None)

    def _cleanup_ipc_socket(self):
        try:
            if os.path.exists(self.ipc_socket):
                os.remove(self.ipc_socket)
        except Exception:
            pass

    def _mpv_ipc_quit(self) -> bool:
        """
        mpv IPC üzerinden "quit" gönder. Çalışırsa True.
        """
        if not self._mpv_alive():
            return False
        if not os.path.exists(self.ipc_socket):
            return False

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.25)
            s.connect(self.ipc_socket)
            s.sendall((json.dumps({"command": ["quit"]}) + "\n").encode("utf-8"))
            s.close()
            return True
        except Exception:
            return False

    def _stop_current(self):
        """
        Kapanırken mpv kalmasın diye:
        1) IPC quit
        2) killpg(SIGTERM)
        3) killpg(SIGKILL)
        """
        if self.proc is None:
            self._cleanup_ipc_socket()
            return

        # 1) IPC quit (en temiz)
        try:
            if self._mpv_ipc_quit():
                for _ in range(20):  # ~1s
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
        except Exception:
            pass

        # 2) hâlâ yaşıyorsa SIGTERM
        try:
            if self.proc.poll() is None:
                self.get_logger().info("Stopping current playback (SIGTERM)...")
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)

                for _ in range(20):  # ~1s
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
        except Exception as e:
            self.get_logger().warn(f"Stop SIGTERM failed: {e}")

        # 3) hâlâ yaşıyorsa SIGKILL
        try:
            if self.proc.poll() is None:
                self.get_logger().warn("mpv did not exit, sending SIGKILL...")
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception as e:
            self.get_logger().warn(f"Stop SIGKILL failed: {e}")

        self.proc = None
        self._cleanup_ipc_socket()

    def _build_base_cmd(self) -> List[str]:
        cmd = [
            self.player,
            "--no-terminal",
            "--really-quiet",
            "--no-video",
            f"--volume={self.volume}",
            f"--input-ipc-server={self.ipc_socket}",
        ]

        if self.alsa_device.strip():
            cmd += ["--ao=alsa", f"--audio-device=alsa/{self.alsa_device.strip()}"]

        if self.extra_args:
            cmd += [str(x) for x in self.extra_args]

        return cmd

    def _spawn(self, cmd: List[str]):
        self._stop_current()
        self._cleanup_ipc_socket()

        self.get_logger().info("Running: " + " ".join(cmd))
        try:
            # ayrı process group => killpg çalışsın
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            self.get_logger().error(f"Player not found: {self.player} (is mpv installed?)")
            self.proc = None
            return
        except Exception as e:
            self.get_logger().error(f"Failed to start player: {e}")
            self.proc = None
            return

        # mpv socket oluşsun
        for _ in range(20):
            if os.path.exists(self.ipc_socket):
                break
            time.sleep(0.05)

        # volume’u garanti et
        self._mpv_set_volume_ipc(self.volume)

    def _mpv_send_ipc(self, payload: dict) -> bool:
        if not self._mpv_alive():
            return False
        if not os.path.exists(self.ipc_socket):
            return False

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.15)
            s.connect(self.ipc_socket)
            s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            try:
                _ = s.recv(4096)
            except Exception:
                pass
            s.close()
            return True
        except Exception:
            return False

    def _mpv_set_volume_ipc(self, vol: int):
        vol = self._clamp_volume(vol)
        ok = self._mpv_send_ipc({"command": ["set_property", "volume", vol]})
        if ok:
            self.get_logger().info(f"Volume set (IPC): {vol}")
        else:
            self.get_logger().info(f"Volume stored (next play): {vol}")

    def _on_ros_shutdown(self):
        try:
            self._stop_current()
        except Exception:
            pass

    # ---------- Callbacks ----------
    def on_path(self, msg: String):
        path = (msg.data or "").strip()
        if not path:
            return
        if not os.path.exists(path):
            self.get_logger().error(f"File not found: {path}")
            return
        cmd = self._build_base_cmd() + [path]
        self._spawn(cmd)

    def on_url(self, msg: String):
        url = (msg.data or "").strip()
        if not url:
            return
        cmd = self._build_base_cmd()
        cmd += ["--ytdl=yes"] if self.ytdl else ["--ytdl=no"]
        cmd += [url]
        self._spawn(cmd)

    def on_stop(self, _msg: Empty):
        self._stop_current()

    def on_volume(self, msg: Int32):
        new_vol = self._clamp_volume(msg.data)
        self.volume = new_vol

        # publish to mobile
        self._publish_volume_state(new_vol)

        if self._mpv_alive():
            self._mpv_set_volume_ipc(new_vol)
        else:
            self.get_logger().info(f"Volume updated (idle): {new_vol}")

    def on_volume_get(self, msg: String):
        _ = (msg.data or "").strip()
        self.get_logger().info("Volume GET requested -> publishing current volume_state")
        self._publish_volume_state(self.volume)


def main():
    rclpy.init()
    node = AudioPlayerNode()

    def _final_cleanup():
        try:
            node._stop_current()
        except Exception:
            pass

    atexit.register(_final_cleanup)

    # ✅ launch SIGINT/SIGTERM gönderince hızlı çık (launch SIGTERM'e yükseltmesin)
    def _handle_signal(_signum, _frame):
        _final_cleanup()
        try:
            node.destroy_node()
        except Exception:
            pass
        # direkt çık: garantili
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _final_cleanup()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()