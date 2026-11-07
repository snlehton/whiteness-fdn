"""Serve web/ on 127.0.0.1 with Cache-Control: no-cache, so the browser revalidates every file and picks up new
exports from scripts/evaluate.py without a hard refresh (plain http.server lets it reuse stale copies).

    python scripts/serve.py [port]
"""
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    handler = functools.partial(NoCacheHandler, directory=str(WEB))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {WEB} at http://127.0.0.1:{port}", flush=True)
        httpd.serve_forever()
