"""A tiny fake Ollama server for engine/client tests."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD = {"description": "Rem looks sad.", "name_text": "Я глупый кролик", "medium": "anime",
        "franchise": [{"name": "Re:Zero", "confidence": "high"}],
        "characters": [{"name": "Rem", "franchise": "Re:Zero", "confidence": "high"}],
        "subjects": ["girl", "Meme"], "meme_format": [], "themes": ["sadness", "x" * 80]}


class FakeOllama:
    def __init__(self):
        self.chat, self.generate, self.verify = [], [], []
        self.models, self.ps, self.pull = ["qwen3-vl:8b"], [], []
        self.answer, self.mode, self.verdicts, self.reject_maxlen = dict(GOOD), "ok", {}, False
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/ps":
                    return self._send(200, {"models": fake.ps})
                self._send(200, {"models": [{"name": n} for n in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate":
                    fake.generate.append(body); return self._send(200, {"done": True})
                if self.path == "/api/show":
                    return self._send(200, {"capabilities": ["completion", "vision"]})
                if self.path == "/api/pull":
                    self.send_response(200); self.end_headers()
                    try:
                        for d in fake.pull:
                            self.wfile.write((json.dumps(d) + "\n").encode()); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                fake.chat.append(body)
                if fake.reject_maxlen and "maxLength" in json.dumps(body.get("format", {})):
                    return self._send(400, {"error": "unsupported schema keyword: maxLength"})
                if "c1" in body["format"]["properties"]:
                    q = body["messages"][1]["content"]; fake.verify.append(q)
                    out = {}
                    for line in q.splitlines()[1:]:
                        key, _, claim = line.partition(": ")
                        out[key] = next((v for k, v in fake.verdicts.items() if k in claim), "unsure")
                    return self._send(200, {"message": {"content": json.dumps(out)}})
                if fake.mode == "think400" and "think" in body:
                    return self._send(400, {"error": "\"qwen\" does not support thinking"})
                if fake.mode == "empty":
                    return self._send(200, {"message": {"content": ""}})
                self._send(200, {"message": {"content": json.dumps(fake.answer)}})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.host = f"127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
