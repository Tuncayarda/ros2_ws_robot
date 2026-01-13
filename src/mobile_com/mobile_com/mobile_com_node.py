from pathlib import Path
import threading
import json

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn


def sanitize_basename(name: str) -> str:
    """
    - Sadece dosya adı (path traversal yok)
    - Güvenli karakterler
    """
    name = Path(name).name.strip()
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " "))
    safe = safe.strip()
    if not safe:
        raise ValueError("invalid name")
    return safe


def normalize_ext(ext: str) -> str:
    ext = (ext or "").strip().lower()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = "." + ext
    return ext


class MobileComNode(Node):
    def __init__(self):
        super().__init__("mobile_com")

        # -------------------- PARAMETRELER --------------------
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8000)

        self.declare_parameter("sounds_dir", str(Path.home() / "Sounds"))
        self.declare_parameter("max_upload_mb", 50)

        self.declare_parameter("emojis_dir", str(Path.home() / "Emojis"))
        self.declare_parameter("max_emoji_upload_mb", 80)

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)

        self.sounds_dir = Path(self.get_parameter("sounds_dir").value)
        self.max_upload_bytes = int(self.get_parameter("max_upload_mb").value) * 1024 * 1024

        self.emojis_dir = Path(self.get_parameter("emojis_dir").value)
        self.max_emoji_upload_bytes = int(self.get_parameter("max_emoji_upload_mb").value) * 1024 * 1024

        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self.emojis_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"Sounds dir: {self.sounds_dir} (max {self.max_upload_bytes/1024/1024:.1f} MB)")
        self.get_logger().info(f"Emojis dir: {self.emojis_dir} (max {self.max_emoji_upload_bytes/1024/1024:.1f} MB)")

        # -------------------- FILE TYPES --------------------
        self.allowed_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        self.allowed_video_exts = {".mp4", ".mov", ".webm"}
        self.allowed_emoji_exts = self.allowed_image_exts | self.allowed_video_exts

        # -------------------- ROS TOPICS --------------------
        # list topics
        self.sounds_list_req_topic = "/mobile/sounds/list_req"
        self.sounds_list_res_topic = "/mobile/sounds/list_res"

        self.emojis_list_req_topic = "/mobile/emojis/list_req"
        self.emojis_list_res_topic = "/mobile/emojis/list_res"

        # delete topics
        self.sounds_delete_req_topic = "/mobile/sounds/delete_req"
        self.sounds_delete_res_topic = "/mobile/sounds/delete_res"

        self.emojis_delete_req_topic = "/mobile/emojis/delete_req"
        self.emojis_delete_res_topic = "/mobile/emojis/delete_res"

        # ROS pubs/subs
        self.pub_sounds_list_res = self.create_publisher(String, self.sounds_list_res_topic, 10)
        self.sub_sounds_list_req = self.create_subscription(String, self.sounds_list_req_topic, self.on_sounds_list_req, 10)

        self.pub_emojis_list_res = self.create_publisher(String, self.emojis_list_res_topic, 10)
        self.sub_emojis_list_req = self.create_subscription(String, self.emojis_list_req_topic, self.on_emojis_list_req, 10)

        self.pub_sounds_delete_res = self.create_publisher(String, self.sounds_delete_res_topic, 10)
        self.sub_sounds_delete_req = self.create_subscription(String, self.sounds_delete_req_topic, self.on_sounds_delete_req, 10)

        self.pub_emojis_delete_res = self.create_publisher(String, self.emojis_delete_res_topic, 10)
        self.sub_emojis_delete_req = self.create_subscription(String, self.emojis_delete_req_topic, self.on_emojis_delete_req, 10)

        # -------------------- FASTAPI --------------------
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"ok": True}

        # ==================== SOUNDS (MP3) ====================
        @app.post("/upload_mp3")
        async def upload_mp3(
            file: UploadFile = File(...),
            name: str = Form(...),
        ):
            try:
                base = sanitize_basename(name)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid name")

            if not base.lower().endswith(".mp3"):
                base = base + ".mp3"

            out_path = self.sounds_dir / base

            if out_path.exists():
                raise HTTPException(status_code=409, detail="file already exists")
            if not file:
                raise HTTPException(status_code=400, detail="missing file")

            written = 0
            try:
                with out_path.open("wb") as f:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self.max_upload_bytes:
                            raise HTTPException(status_code=413, detail="file too large")
                        f.write(chunk)
            except HTTPException:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: pass
                raise
            except Exception as e:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: pass
                raise HTTPException(status_code=500, detail=f"save failed: {e}")

            self.get_logger().info(f"Saved mp3: {out_path} ({written/1024/1024:.2f} MB)")

            # ✅ yeni ses -> listeyi broadcast
            self.broadcast_sounds_list()

            return {"ok": True, "filename": base, "size_mb": round(written / 1024 / 1024, 2)}

        # ==================== EMOJIS (PHOTO/VIDEO) ====================
        @app.post("/upload_emoji")
        async def upload_emoji(
            file: UploadFile = File(...),
            name: str = Form(...),
        ):
            if not file:
                raise HTTPException(status_code=400, detail="missing file")

            try:
                base = sanitize_basename(name)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid name")

            # uzantıyı upload edilen dosyadan al
            orig = Path(file.filename or "").name
            ext = normalize_ext(Path(orig).suffix)

            if ext not in self.allowed_emoji_exts:
                raise HTTPException(
                    status_code=400,
                    detail=f"invalid file type. allowed: {sorted(self.allowed_emoji_exts)}"
                )

            base_no_ext = base
            if Path(base).suffix:
                base_no_ext = Path(base).stem

            out_name = f"{base_no_ext}{ext}"
            out_path = self.emojis_dir / out_name

            if out_path.exists():
                raise HTTPException(status_code=409, detail="file already exists")

            written = 0
            try:
                with out_path.open("wb") as f:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self.max_emoji_upload_bytes:
                            raise HTTPException(status_code=413, detail="file too large")
                        f.write(chunk)
            except HTTPException:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: pass
                raise
            except Exception as e:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: pass
                raise HTTPException(status_code=500, detail=f"save failed: {e}")

            kind = "video" if ext in self.allowed_video_exts else "image"
            self.get_logger().info(f"Saved emoji ({kind}): {out_path} ({written/1024/1024:.2f} MB)")

            # ✅ yeni emoji -> listeyi broadcast
            self.broadcast_emojis_list()

            return {"ok": True, "filename": out_name, "type": kind, "size_mb": round(written / 1024 / 1024, 2)}

        self._app = app

    # -------------------- SOUNDS: payload + broadcast --------------------
    def _build_sounds_payload(self) -> dict:
        items = []
        for p in sorted(self.sounds_dir.glob("*.mp3")):
            try:
                st = p.stat()
                items.append({"name": p.name, "size_bytes": st.st_size})
            except Exception:
                continue
        return {"ok": True, "count": len(items), "items": items}

    def broadcast_sounds_list(self):
        try:
            payload = self._build_sounds_payload()
            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.pub_sounds_list_res.publish(out)
            self.get_logger().info(f"Broadcast sounds list: {payload['count']} items")
        except Exception as e:
            out = String()
            out.data = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            self.pub_sounds_list_res.publish(out)
            self.get_logger().error(f"Sounds broadcast failed: {e}")

    def on_sounds_list_req(self, _msg: String):
        self.broadcast_sounds_list()

    # -------------------- EMOJIS: payload + broadcast --------------------
    def _build_emojis_payload(self) -> dict:
        items = []
        for p in sorted(self.emojis_dir.iterdir()):
            if not p.is_file():
                continue
            ext = normalize_ext(p.suffix)
            if ext not in self.allowed_emoji_exts:
                continue
            try:
                st = p.stat()
                kind = "video" if ext in self.allowed_video_exts else "image"
                items.append({
                    "name": p.name,
                    "type": kind,
                    "ext": ext,
                    "size_bytes": st.st_size,
                })
            except Exception:
                continue
        return {"ok": True, "count": len(items), "items": items}

    def broadcast_emojis_list(self):
        try:
            payload = self._build_emojis_payload()
            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.pub_emojis_list_res.publish(out)
            self.get_logger().info(f"Broadcast emojis list: {payload['count']} items")
        except Exception as e:
            out = String()
            out.data = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            self.pub_emojis_list_res.publish(out)
            self.get_logger().error(f"Emojis broadcast failed: {e}")

    def on_emojis_list_req(self, _msg: String):
        self.broadcast_emojis_list()

    # -------------------- DELETE: SOUNDS --------------------
    def on_sounds_delete_req(self, msg: String):
        """
        msg.data JSON string:
          {"name":"file.mp3"}
        """
        try:
            data = json.loads(msg.data)
            name = sanitize_basename(data.get("name", ""))

            if not name.lower().endswith(".mp3"):
                raise ValueError("sound must be .mp3")

            path = self.sounds_dir / name
            if not path.exists():
                raise FileNotFoundError("file not found")

            path.unlink()
            self.get_logger().info(f"Deleted sound: {name}")

            # ✅ Silince yeni listeyi broadcast et
            self.broadcast_sounds_list()

            res = {"ok": True, "name": name}
        except Exception as e:
            res = {"ok": False, "error": str(e)}

        out = String()
        out.data = json.dumps(res, ensure_ascii=False)
        self.pub_sounds_delete_res.publish(out)

    # -------------------- DELETE: EMOJIS --------------------
    def on_emojis_delete_req(self, msg: String):
        """
        msg.data JSON string:
          {"name":"happy.png"}  (uzantı şart)
        """
        try:
            data = json.loads(msg.data)
            name = sanitize_basename(data.get("name", ""))

            ext = normalize_ext(Path(name).suffix)
            if ext not in self.allowed_emoji_exts:
                raise ValueError(f"invalid emoji ext: {ext}")

            path = self.emojis_dir / name
            if not path.exists():
                raise FileNotFoundError("file not found")

            path.unlink()
            self.get_logger().info(f"Deleted emoji: {name}")

            # ✅ Silince yeni listeyi broadcast et
            self.broadcast_emojis_list()

            res = {"ok": True, "name": name}
        except Exception as e:
            res = {"ok": False, "error": str(e)}

        out = String()
        out.data = json.dumps(res, ensure_ascii=False)
        self.pub_emojis_delete_res.publish(out)

    # -------------------- HTTP server --------------------
    def run_http(self):
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")


def main():
    rclpy.init()
    node = MobileComNode()

    http_thread = threading.Thread(target=node.run_http, daemon=True)
    http_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()