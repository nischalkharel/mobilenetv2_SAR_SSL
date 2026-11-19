"""Local web app for the results tracker: serves tracker/web, the collected JSON, repository images, and the idea box.

    python tracker/serve.py                # collects first, then opens http://localhost:8765 in the default browser
    python tracker/serve.py --no-collect   # skip the collection step
    python tracker/serve.py --port 9000 --no-browser

Routes:
    /                      tracker/web/index.html (and every other file in tracker/web)
    /data/<name>.json      tracker/data/<name>.json written by collect.py
    /file/<repo path>      a png / jpg / csv / md / json / txt file from inside the repository (read-only)
    /api/ideas             GET list, POST add {text, tags, who}; /api/ideas/<id> PUT update fields, DELETE remove
    /api/collect           POST: re-run collect.py and return its output
Ideas live in tracker/ideas.json (committed with the repository, so they travel with it).
Standard library only; nothing to install.
"""
import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "tracker" / "web"
DATA = REPO / "tracker" / "data"
IDEAS = REPO / "tracker" / "ideas.json"
FILE_TYPES = {".png", ".jpg", ".jpeg", ".svg", ".csv", ".md", ".json", ".txt"}
LOCK = threading.Lock()


def load_ideas():
    if not IDEAS.exists():
        return {"ideas": [], "next_id": 1}
    return json.loads(IDEAS.read_text(encoding="utf-8"))


def save_ideas(d):
    tmp = IDEAS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(IDEAS)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):       # quieter log: one line per request without the timestamp noise
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    # ---- helpers ----
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        if not path.is_file():
            return self.send_error(404, f"not found: {path.name}")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".md":
            ctype = "text/plain; charset=utf-8"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def repo_file(self, relpath):
        p = (REPO / unquote(relpath)).resolve()
        if REPO not in p.parents or p.suffix.lower() not in FILE_TYPES:
            return None
        return p

    # ---- routes ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        if path.startswith("/data/"):
            return self.send_file(DATA / path[6:])
        if path.startswith("/file/"):
            p = self.repo_file(path[6:])
            return self.send_file(p) if p else self.send_error(403, "only images and text files inside the repository")
        if path == "/api/ideas":
            with LOCK:
                return self.send_json(load_ideas())
        p = (WEB / path.lstrip("/")).resolve()
        if WEB not in p.parents:
            return self.send_error(403)
        return self.send_file(p)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/ideas":
            b = self.body_json()
            text = (b.get("text") or "").strip()
            if not text:
                return self.send_json({"error": "empty idea"}, 400)
            with LOCK:
                d = load_ideas()
                idea = {"id": d["next_id"], "created": datetime.now().isoformat(timespec="minutes"), "who": b.get("who") or "Nischal",
                        "text": text, "tags": [t.strip() for t in (b.get("tags") or []) if t.strip()], "status": "open", "replies": []}
                d["ideas"].append(idea)
                d["next_id"] += 1
                save_ideas(d)
            return self.send_json(idea, 201)
        if path == "/api/collect":
            r = subprocess.run([sys.executable, str(REPO / "tracker" / "collect.py")], capture_output=True, text=True, cwd=REPO)
            return self.send_json({"ok": r.returncode == 0, "output": r.stdout + r.stderr})
        return self.send_error(404)

    def do_PUT(self):
        path = urlparse(self.path).path
        if path.startswith("/api/ideas/"):
            b = self.body_json()
            with LOCK:
                d = load_ideas()
                for idea in d["ideas"]:
                    if str(idea["id"]) == path.rsplit("/", 1)[1]:
                        for k in ("text", "tags", "status"):
                            if k in b:
                                idea[k] = b[k]
                        if b.get("reply"):
                            idea["replies"].append({"who": b.get("who") or "Nischal", "when": datetime.now().isoformat(timespec="minutes"), "text": b["reply"].strip()})
                        save_ideas(d)
                        return self.send_json(idea)
            return self.send_json({"error": "no such idea"}, 404)
        return self.send_error(404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/ideas/"):
            with LOCK:
                d = load_ideas()
                before = len(d["ideas"])
                d["ideas"] = [i for i in d["ideas"] if str(i["id"]) != path.rsplit("/", 1)[1]]
                save_ideas(d)
            return self.send_json({"removed": before - len(d["ideas"])})
        return self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-collect", action="store_true", help="do not run collect.py before serving")
    a = ap.parse_args()
    if not a.no_collect:
        subprocess.run([sys.executable, str(REPO / "tracker" / "collect.py")], check=True, cwd=REPO)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    print(f"tracker at {url}  (Ctrl-C stops it)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
