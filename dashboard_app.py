#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sys
import urllib.parse

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CLIPS = [
    {
        "id": "clip_1",
        "name": "Stirring Sauce (Known Class)",
        "filename": "stirring_sauce.mp4",
        "is_novel": False,
        "nearest_class": "Stirring",
        "vacuity": 0.18,
        "distance": 1.2,
        "x": 1.5,
        "y": -0.8
    },
    {
        "id": "clip_2",
        "name": "Slicing Onion (Known Class)",
        "filename": "slicing_onion.mp4",
        "is_novel": False,
        "nearest_class": "Slicing",
        "vacuity": 0.22,
        "distance": 1.5,
        "x": -2.1,
        "y": 1.4
    },
    {
        "id": "clip_3",
        "name": "Peeling Mango (Novel Class - Discovery Flow)",
        "filename": "peeling_mango.mp4",
        "is_novel": True,
        "nearest_class": "Slicing",
        "vacuity": 0.89,
        "distance": 4.8,
        "x": 3.8,
        "y": 3.2
    }
]

CENTROIDS = [
    {"name": "Stirring", "x": 1.2, "y": -0.9, "color": "#10b981"},  # Emerald
    {"name": "Slicing", "x": -1.9, "y": 1.2, "color": "#3b82f6"},   # Blue
    {"name": "Washing", "x": -0.5, "y": -2.0, "color": "#a855f7"}   # Purple
]

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Prevent accessing files outside current directory
        normalized = os.path.normpath(path)
        return super().translate_path(normalized)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            index_path = os.path.join(DIRECTORY, "templates", "index.html")
            with open(index_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if path == "/api/clips":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"clips": CLIPS, "centroids": CENTROIDS}).encode())
            return

        if path == "/api/analyze":
            query = urllib.parse.parse_qs(parsed_url.query)
            clip_id = query.get("clip_id", [None])[0]
            clip = next((c for c in CLIPS if c["id"] == clip_id), None)
            
            if not clip:
                self.send_error(404, "Clip not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "clip": clip,
                "centroids": CENTROIDS,
                "threshold": 0.45
            }).encode())
            return

        # Serve video files from demo_clips directory
        if path.startswith("/video/"):
            filename = path[7:]
            filepath = os.path.join(DIRECTORY, "demo_clips", filename)
            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Video file not found")
                return

        return super().do_GET()

def main():
    # Make sure templates and demo_clips exist
    os.makedirs(os.path.join(DIRECTORY, "templates"), exist_ok=True)
    os.makedirs(os.path.join(DIRECTORY, "demo_clips"), exist_ok=True)
    
    server_address = ("", PORT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, DashboardHandler) as httpd:
        print(f"Action Discovery Dashboard running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")
            sys.exit(0)

if __name__ == "__main__":
    main()
