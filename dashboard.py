"""Local web dashboard and section-based monitoring state."""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import urlsplit

import cv2


SECTIONS = ("A", "B")
APPEARANCE_CUE_TITLES = {
    "possible_dark_clothing": "Possible dark clothing (visual cue)",
    "possible_face_covering": "Possible face covering (visual cue)",
}


def _normalize_name(name: str) -> str:
    return " ".join(name.casefold().split())


class DashboardState:
    def __init__(self, dataset_path: str) -> None:
        self.config_path = Path(dataset_path) / "classroom.json"
        self._lock = threading.RLock()
        self._room_section = "A"
        self._members: Dict[str, str] = {}
        self._registered_names: List[str] = []
        self._detections: List[dict] = []
        self._events: List[dict] = []
        self._active_alerts = set()
        self._last_event_at: Dict[str, float] = {}
        self._next_event_id = 1
        self._fps = 0.0
        self._jpeg: Optional[bytes] = None
        self._load_config()

    def _load_config(self) -> None:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Cannot load classroom settings: {exc}")
            return

        room = config.get("room_section")
        if room in SECTIONS:
            self._room_section = room
        members = config.get("members", {})
        if isinstance(members, dict):
            self._members = {
                _normalize_name(name): section
                for name, section in members.items()
                if isinstance(name, str) and section in SECTIONS
            }

    def update_config(self, config: dict) -> None:
        room = config.get("room_section")
        members = config.get("members")
        if room not in SECTIONS or not isinstance(members, dict):
            raise ValueError("Choose room section A or B and provide a roster.")

        with self._lock:
            registered = {_normalize_name(name): name for name in self._registered_names}
            updated = {}
            for name, section in members.items():
                normalized = _normalize_name(name) if isinstance(name, str) else ""
                if normalized not in registered:
                    continue
                if section in SECTIONS:
                    updated[normalized] = section
                elif section not in (None, ""):
                    raise ValueError("Roster sections must be A, B, or unassigned.")

            self._room_section = room
            self._members = updated
            saved = {
                "room_section": room,
                "members": {
                    registered[name]: section for name, section in updated.items()
                },
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.config_path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            temp_path.replace(self.config_path)

    def update(self, detections: List[dict], registered_names: List[str], fps: float) -> None:
        with self._lock:
            self._registered_names = list(registered_names)
            self._fps = fps
            current_alerts = set()
            visible = []

            for item in detections:
                name = item["name"]
                recognized = bool(item["recognized"])
                appearance_cues = [
                    cue for cue in item.get("appearance_cues", [])
                    if cue in APPEARANCE_CUE_TITLES
                ]
                section = self._members.get(_normalize_name(name)) if recognized else None
                identity_alert = None
                if not recognized:
                    identity_alert = "unknown"
                    title = "Unregistered person"
                elif section is not None and section != self._room_section:
                    identity_alert = "section_mismatch"
                    title = f"Section {section} student in Section {self._room_section}"

                alert_types = []
                if identity_alert:
                    alert_types.append((identity_alert, title))
                alert_types.extend(
                    (cue, APPEARANCE_CUE_TITLES[cue]) for cue in appearance_cues
                )
                for alert_type, title in alert_types:
                    alert_id = f"{item['track_id']}:{alert_type}"
                    current_alerts.add(alert_id)
                    if alert_id not in self._active_alerts:
                        event_key = (
                            f"{alert_type}:{_normalize_name(name) if recognized else 'unknown'}"
                            if identity_alert == alert_type
                            else (
                                f"appearance:{alert_type}:"
                                f"{_normalize_name(name) if recognized else item['track_id']}"
                            )
                        )
                        now = time.monotonic()
                        if now - self._last_event_at.get(event_key, 0.0) >= 20.0:
                            self._events.insert(0, {
                                "id": self._next_event_id,
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "name": name if recognized else "Unknown face",
                                "type": (
                                    alert_type if alert_type in ("unknown", "section_mismatch")
                                    else "appearance"
                                ),
                                "title": title,
                            })
                            self._last_event_at[event_key] = now
                            self._next_event_id += 1

                visible.append({
                    "track_id": item["track_id"],
                    "name": name,
                    "section": section,
                    "recognized": recognized,
                    "score": round(float(item["score"]), 3),
                    "speed": round(float(item["speed"]), 2),
                    "status": item["status"],
                    "alert": identity_alert or ("appearance" if appearance_cues else None),
                    "appearance_cues": appearance_cues,
                })

            self._active_alerts = current_alerts
            self._detections = visible
            self._events = self._events[:40]

    def set_frame(self, frame) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()

    def frame(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            people = [
                {"name": name, "section": self._members.get(_normalize_name(name))}
                for name in self._registered_names
            ]
            return {
                "room_section": self._room_section,
                "registered_count": len(self._registered_names),
                "fps": round(self._fps, 1),
                "detections": list(self._detections),
                "people": people,
                "events": list(self._events),
                "active_alert_count": len(self._active_alerts),
                "frame_available": self._jpeg is not None,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()


class DashboardServer:
    def __init__(self, state: DashboardState, host: str = "127.0.0.1", port: int = 8765) -> None:
        html_path = Path(__file__).with_name("dashboard.html")
        page = html_path.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def _send_json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif path == "/api/state":
                    self._send_json(state.snapshot())
                elif path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            jpeg = state.frame()
                            if jpeg:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                    + str(len(jpeg)).encode("ascii")
                                    + b"\r\n\r\n" + jpeg + b"\r\n"
                                )
                                self.wfile.flush()
                            time.sleep(0.12)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                try:
                    path = urlsplit(self.path).path
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 65536:
                        self._send_json({"error": "Request too large."}, 413)
                        return
                    body = self.rfile.read(length)
                    payload = json.loads(body) if body else {}
                    if path == "/api/config":
                        state.update_config(payload)
                        self._send_json({"ok": True})
                    elif path == "/api/events/clear":
                        state.clear_events()
                        self._send_json({"ok": True})
                    else:
                        self.send_error(404)
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"error": str(exc)}, 400)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.url = f"http://{host}:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> str:
        self._thread.start()
        return self.url

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()