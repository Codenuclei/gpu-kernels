import os
import sys
import json
import time
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from engine import KernelCrucibleEngine
from rlvr_grpo_engine import RLVREngine

engine = KernelCrucibleEngine()
rlvr_engine = RLVREngine(group_size=4, baseline_latency_us=engine.gym_env.target_baseline_us)
auto_running = False
auto_thread = None

def auto_loop():
    global auto_running
    while auto_running:
        try:
            engine.step_arena()
            time.sleep(1.5)
        except Exception as e:
            print("Auto loop error:", e)
            time.sleep(2)

class CrucibleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(BASE_DIR, "static"), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "champion": engine.champion,
                "challenger": engine.last_challenger,
                "telemetry": engine.get_gpu_telemetry(),
                "hall_of_fame": engine.hall_of_fame[:8],
                "round_num": engine.round_num,
                "generation": engine.generation,
                "auto_running": auto_running,
                "log": engine.event_log[:20],
                "last_grpo_cohort": getattr(engine, "last_grpo_cohort", None),
                "baseline_latency_us": engine.gym_env.target_baseline_us
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif parsed.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    data = {
                        "champion": engine.champion,
                        "challenger": engine.last_challenger,
                        "telemetry": engine.get_gpu_telemetry(),
                        "hall_of_fame": engine.hall_of_fame[:8],
                        "round_num": engine.round_num,
                        "generation": engine.generation,
                        "auto_running": auto_running,
                        "log": engine.event_log[:20],
                        "last_grpo_cohort": getattr(engine, "last_grpo_cohort", None),
                        "baseline_latency_us": engine.gym_env.target_baseline_us
                    }
                    msg = f"data: {json.dumps(data)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0)
            except Exception:
                pass
        else:
            super().do_GET()

    def do_POST(self):
        global auto_running, auto_thread
        parsed = urlparse(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if parsed.path == "/api/step":
            res = engine.step_arena()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        elif parsed.path == "/api/grpo_step":
            res = engine.step_grpo(rlvr_engine)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        elif parsed.path == "/api/start_auto":
            if not auto_running:
                auto_running = True
                auto_thread = threading.Thread(target=auto_loop, daemon=True)
                auto_thread.start()
            self.wfile.write(json.dumps({"status": "auto_started"}).encode("utf-8"))
        elif parsed.path == "/api/stop_auto":
            auto_running = False
            self.wfile.write(json.dumps({"status": "auto_stopped"}).encode("utf-8"))
        else:
            self.wfile.write(json.dumps({"error": "not_found"}).encode("utf-8"))

def run_server(port=5050):
    server = ThreadingHTTPServer(("0.0.0.0", port), CrucibleHandler)
    print(f"🔥 Kernel Crucible Coliseum UI live at http://127.0.0.1:{port}")
    server.serve_forever()

if __name__ == "__main__":
    port = 5050
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
