#!/usr/bin/env python3
import os
import signal
import subprocess
import json
import socket
import time
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
        self.declare_parameter("topic_volume", "/audio/volume")   # ✅ NEW

        self.declare_parameter("player", "mpv")
        self.declare_parameter("volume", 30)          # 0..100
        self.declare_parameter("ytdl", True)          # use mpv --ytdl for url
        self.declare_parameter("alsa_device", "")     # e.g. "default" or "hw:0,0"
        self.declare_parameter("extra_args", [])      # list[str]

        # mpv IPC
        self.declare_parameter("ipc_socket", "/tmp/mpv_audio.sock")  # ✅ NEW

        self.topic_path   = self.get_parameter("topic_path").value
        self.topic_url    = self.get_parameter("topic_url").value
        self.topic_stop   = self.get_parameter("topic_stop").value
        self.topic_volume = self.get_parameter("topic_volume").value

        self.player = str(self.get_parameter("player").value)
        self.volume = int(self.get_parameter("volume").value)
        self.ytdl   = bool(self.get_parameter("ytdl").value)
        self.alsa_device = str(self.get_parameter("alsa_device").value)
        self.extra_args: List[str] = list(self.get_parameter("extra_args").value)

        self.ipc_socket = str(self.get_parameter("ipc_socket").value)

        self.volume = self._clamp_volume(self.volume)

        # Current playback process
        self.proc: Optional[subprocess.Popen] = None

        # -------- Subscriptions --------
        self.sub_path = self.create_subscription(String, self.topic_path, self.on_path, 10)
        self.sub_url  = self.create_subscription(String, self.topic_url,  self.on_url,  10)
        self.sub_stop = self.create_subscription(Empty,  self.topic_stop, self.on_stop, 10)
        self.sub_vol  = self.create_subscription(Int32,  self.topic_volume, self.on_volume, 10)  # ✅ NEW

        self.get_logger().info(
            f"AudioPlayer ready | path={self.topic_path} url={self.topic_url} stop={self.topic_stop} vol={self.topic_volume} "
            f"| player={self.player} volume={self.volume} ytdl={int(self.ytdl)} alsa_device='{self.alsa_device}' ipc='{self.ipc_socket}'"
        )

    # ---------- Helpers ----------
    @staticmethod
    def _clamp_volume(v: int) -> int:
        return max(0, min(100, int(v)))

    def _mpv_alive(self) -> bool:
        return self.proc is not None and (self.proc.poll() is None)

    def _stop_current(self):
        if self.proc is None:
            return

        try:
            if self.proc.poll() is None:
                self.get_logger().info("Stopping current playback...")
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception as e:
            self.get_logger().warn(f"Stop failed: {e}")

        self.proc = None

        # cleanup socket file (safe)
        try:
            if os.path.exists(self.ipc_socket):
                os.remove(self.ipc_socket)
        except Exception:
            pass

    def _build_base_cmd(self) -> List[str]:
        # IMPORTANT: IPC enabled so we can change volume live
        cmd = [
            self.player,
            "--no-terminal",
            "--really-quiet",
            "--no-video",
            f"--volume={self.volume}",
            f"--input-ipc-server={self.ipc_socket}",   # ✅ NEW
        ]

        # ALSA output (optional)
        if self.alsa_device.strip():
            cmd += ["--ao=alsa", f"--audio-device=alsa/{self.alsa_device.strip()}"]

        if self.extra_args:
            cmd += [str(x) for x in self.extra_args]

        return cmd

    def _spawn(self, cmd: List[str]):
        self._stop_current()

        # remove old socket before starting mpv
        try:
            if os.path.exists(self.ipc_socket):
                os.remove(self.ipc_socket)
        except Exception:
            pass

        self.get_logger().info("Running: " + " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            self.get_logger().error(f"Player not found: {self.player} (is mpv installed?)")
            self.proc = None
        except Exception as e:
            self.get_logger().error(f"Failed to start player: {e}")
            self.proc = None
            return

        # give mpv a moment to create the socket
        for _ in range(20):
            if os.path.exists(self.ipc_socket):
                break
            time.sleep(0.05)

        # apply current volume via IPC too (guaranteed)
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
            # read optional reply (not strictly required)
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
            # if mpv not running, just update default volume for next spawn
            self.get_logger().info(f"Volume stored (next play): {vol}")

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

        # if playing => change live, else just store
        if self._mpv_alive():
            self._mpv_set_volume_ipc(new_vol)
        else:
            self.get_logger().info(f"Volume updated (idle): {new_vol}")


def main():
    rclpy.init()
    node = AudioPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_current()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()