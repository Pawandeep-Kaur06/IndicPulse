"""
IndicPulse Web Server
Serves the clean text-intelligence playground and provides live API endpoints
for Language Identification, Emotion Recognition, and Multi-Label Intent Classification.

Usage:
    python app.py
    Open http://localhost:8000 in your browser.
"""

import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import IndicPulsePipeline

PUBLIC_DIR = ROOT_DIR / "public"
PORT = 8000

# Global pipeline instance (loaded once on server start)
print("=" * 60)
print("Initializing IndicPulse Multi-Module Intelligence Pipeline...")
print("=" * 60)
pipeline = IndicPulsePipeline()
print("Pipeline loaded successfully! All 3 modules ready.\n")


class IndicPulseHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for IndicPulse API and web interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        if self.path in {"/api/status", "/status"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = {
                "status": "online",
                "modules": {
                    "language": "IndicCMix TF-IDF (99.7% Accuracy)",
                    "emotion": "EmoHinD-58K Baseline (64.7% Accuracy)",
                    "intent": "Fine-Tuned XLM-RoBERTa (Calibrated)",
                },
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        if self.path in {"/api/predict", "/predict"}:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                text = data.get("message") or data.get("text") or ""
                if not text:
                    self.send_response(HTTPStatus.BAD_REQUEST)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Empty message"}).encode("utf-8"))
                    return

                # Run unified pipeline with detailed scores
                result = pipeline.predict(text, detailed=True)

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def main():
    server_address = ("", PORT)
    httpd = ThreadingHTTPServer(server_address, IndicPulseHandler)
    print("=" * 60)
    print(f"IndicPulse Web Server is live at:")
    print(f"  --> http://localhost:{PORT}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down IndicPulse server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
