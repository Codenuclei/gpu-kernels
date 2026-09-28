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
from test_suites import WORKLOAD_TEMPLATES
import patch_unsloth

engine = KernelCrucibleEngine()
rlvr_engine = RLVREngine(group_size=4, baseline_latency_us=engine.gym_env.target_baseline_us)
auto_running = False
auto_thread = None

def auto_loop():
    global auto_running
    while auto_running:
        try:
            # Dedicated RLVR + GRPO Parallel Cohort Tournament (G=4 Slots Saturated)
            engine.step_grpo(rlvr_engine)
            time.sleep(1.0)
        except Exception as e:
            print("Auto RLVR loop error:", e)
            time.sleep(2)

def build_status_payload():
    return {
        "champion": engine.champion,
        "challenger": engine.last_challenger,
        "telemetry": engine.get_gpu_telemetry(),
        "hall_of_fame": engine.hall_of_fame[:8],
        "round_num": engine.round_num,
        "generation": engine.generation,
        "auto_running": auto_running,
        "log": engine.event_log[:20],
        "last_grpo_cohort": getattr(engine, "last_grpo_cohort", None),
        "baseline_latency_us": engine.gym_env.target_baseline_us,
        "unsloth_patched": patch_unsloth.is_patched(),
        "saturation_detected": getattr(engine, "saturation_detected", False),
        "rounds_without_ascension": getattr(engine, "rounds_without_ascension", 0)
    }

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
            self.wfile.write(json.dumps(build_status_payload()).encode("utf-8"))
        elif parsed.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    msg = f"data: {json.dumps(build_status_payload())}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0)
            except Exception:
                pass
        elif parsed.path == "/api/workloads":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "current": engine.current_workload,
                "workloads": WORKLOAD_TEMPLATES
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif parsed.path == "/api/unsloth_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "is_patched": patch_unsloth.is_patched(),
                "champion": engine.champion["name"],
                "latency_us": engine.champion["latency_us"]
            }).encode("utf-8"))
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
        elif parsed.path == "/api/set_workload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            req = json.loads(body)
            w_key = req.get("workload", "split_kv_d64")
            engine.set_workload(w_key)
            self.wfile.write(json.dumps({"status": "workload_set", "workload": w_key}).encode("utf-8"))
        elif parsed.path == "/api/start_auto":
            if not auto_running:
                auto_running = True
                auto_thread = threading.Thread(target=auto_loop, daemon=True)
                auto_thread.start()
            self.wfile.write(json.dumps({"status": "auto_started"}).encode("utf-8"))
        elif parsed.path == "/api/stop_auto":
            auto_running = False
            self.wfile.write(json.dumps({"status": "auto_stopped"}).encode("utf-8"))
        elif parsed.path == "/api/unsloth_patch":
            patch_unsloth.patch_unsloth()
            self.wfile.write(json.dumps({"status": "patched", "is_patched": True}).encode("utf-8"))
        elif parsed.path == "/api/unsloth_unpatch":
            patch_unsloth.unpatch_unsloth()
            self.wfile.write(json.dumps({"status": "unpatched", "is_patched": False}).encode("utf-8"))
        elif parsed.path == "/api/trigger_post_train":
            import auto_post_train
            dataset_path = auto_post_train.export_gold_dataset(engine.hall_of_fame)
            success = auto_post_train.run_unsloth_sft(dataset_path)
            engine.event_log.insert(0, "🎓 [POST-TRAIN] Distilled Hall of Fame into serving model weights!")
            self.wfile.write(json.dumps({"status": "post_trained", "success": success}).encode("utf-8"))
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
