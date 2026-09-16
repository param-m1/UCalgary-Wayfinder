"""
Serve the wayfinder locally.

    python3 serve.py [port]      then open http://localhost:8770/app/

Two small things it does beyond http.server's default:

* Serves navigation-graph.json.gz in place of the .json when the browser
  accepts gzip. The graph is repetitive JSON that compresses to about a tenth
  of its size; a real static host does this for you, http.server does not.
* Takes its root from the script's own location rather than the working
  directory, because os.getcwd() is not always readable under macOS's file
  access controls and the stock server dies on it at import time.
"""
import functools
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        if self.path.endswith(".json") and "gzip" in self.headers.get("Accept-Encoding", ""):
            gz = self.translate_path(self.path) + ".gz"
            if os.path.exists(gz):
                try:
                    fh = open(gz, "rb")
                except OSError:
                    return super().send_head()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(os.fstat(fh.fileno()).st_size))
                self.end_headers()
                return fh
        return super().send_head()

    def log_message(self, *args):
        pass


print(f"serving {ROOT} -> http://localhost:{PORT}/app/")
http.server.ThreadingHTTPServer(("127.0.0.1", PORT),
                                functools.partial(Handler, directory=ROOT)).serve_forever()
