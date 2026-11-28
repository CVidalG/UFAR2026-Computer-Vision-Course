"""Test of labelstudio_yolo.py, as a user runs it (GitHub Actions: Ubuntu, Windows, macOS; or by hand):

    uv run --no-project python test_launcher.py path/to/images

The launcher of this folder is started (first run: installation), then checked: Label Studio answers; the ML backend
asks for its account and answers with it; the project has the tasks of the images; the backend predicts for the first
task; the dashboard asks for its key and shows the page with it. Then Ctrl-C: the launcher prints its summary and stops
both servers. Exit code 0 when everything passed; the log of the launcher is printed otherwise.
"""
import base64
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILES = r"(?i).*\.(jpe?g|png|gif|bmp|webp)$"
PORT, ML_PORT = 8181, 9191


def main():
    images = os.path.abspath(sys.argv[1])
    log_file = os.path.join(HERE, "data", "test-launcher.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    checks = {}

    def text():
        with open(log_file, errors="replace") as f:
            return re.sub(r"\x1b\[[0-9;]*m", "", f.read())

    def check(name, ok, detail=""):
        checks[name] = "ok" if ok else f"FAILED {detail}".strip()

    # The launcher, in this console (Windows: Ctrl-C reaches it; its servers are in their own process groups),
    # in its own session elsewhere (Ctrl-C sent to it only)
    options = {} if os.name == "nt" else {"start_new_session": True}
    with open(log_file, "w") as out:
        launcher = subprocess.Popen(["uv", "run", "labelstudio_yolo.py", images, "--no-browser", "--dashboard",
                                     "--port", str(PORT), "--ml-port", str(ML_PORT)],
                                    cwd=HERE, stdout=out, stderr=subprocess.STDOUT, **options)
    start = time.time()
    while "Stop           Ctrl-C" not in text() and launcher.poll() is None and time.time() - start < 2400:
        time.sleep(5)
    check("banner", "Stop           Ctrl-C" in text(), f"after {time.time() - start:.0f} s")
    print(f"Launcher ready after {time.time() - start:.0f} s" if checks["banner"] == "ok" else "Launcher not ready",
          flush=True)
    dashboard_port = None
    try:
        if checks["banner"] == "ok":
            settings = json.load(open(os.path.join(HERE, "data", "settings.json")))

            def api(method, path, body=None):
                request = urllib.request.Request(f"http://localhost:{settings['port']}" + path, method=method,
                                                 data=None if body is None else json.dumps(body).encode(),
                                                 headers={"Authorization": "Token " + settings["token"],
                                                          "Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=900) as r:
                    content = r.read()
                return json.loads(content) if content else None

            def status(url, headers=None):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60) as r:
                        return r.status
                except urllib.error.HTTPError as e:
                    return e.code

            check("Label Studio /health", status(f"http://localhost:{settings['port']}/health") == 200)
            account = {"Authorization": "Basic " + base64.b64encode(
                f"{settings['ml_user']}:{settings['ml_password']}".encode()).decode()}
            check("ML backend without its account: 401",
                  status(f"http://localhost:{settings['ml_port']}/health") == 401)
            check("ML backend with its account: 200",
                  status(f"http://localhost:{settings['ml_port']}/health", account) == 200)
            pid = int(re.search(r"/projects/(\d+)/data", text()).group(1))
            expected = sum(1 for name in os.listdir(images) if re.match(IMAGE_FILES, name))
            tasks = api("GET", f"/api/tasks?project={pid}&page_size=100")["tasks"]
            check("tasks of the images", len(tasks) == expected, f"{len(tasks)} tasks, {expected} images")
            api("POST", f"/api/dm/actions?id=retrieve_tasks_predictions&project={pid}",
                {"selectedItems": {"all": False, "included": [tasks[0]["id"]]}})
            predictions = api("GET", f"/api/predictions?task={tasks[0]['id']}")
            check("prediction of the backend", len(predictions) > 0)
            tabs = [v["data"].get("title") for v in api("GET", f"/api/dm/views/?project={pid}")]
            check("tab To annotate", "To annotate" in tabs, str(tabs))
            found = re.search(r"Dashboard      http://[^:/]+:(\d+)/\?key=(\S+)", text())
            dashboard_port, key = found.group(1), found.group(2)

            def get(path, cookie=""):
                connection = http.client.HTTPConnection("127.0.0.1", int(dashboard_port), timeout=120)
                connection.request("GET", path, headers={"Host": f"localhost:{dashboard_port}", "Cookie": cookie})
                response = connection.getresponse()
                return response.status, response.getheader("Set-Cookie") or "", response.read()

            check("dashboard without its key: 403", get("/")[0] == 403)
            code, cookie, _ = get(f"/?key={key}")
            code, _, page = get("/", cookie.split(";")[0])
            check("dashboard with its key", code == 200 and b"To annotate first" in page, str(code))
    finally:
        # Ctrl-C (Windows: to this console, the test ignoring it)
        if launcher.poll() is None:
            if os.name == "nt":
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                os.kill(0, signal.CTRL_C_EVENT)
            else:
                os.killpg(launcher.pid, signal.SIGINT)
            try:
                launcher.wait(timeout=120)
            except subprocess.TimeoutExpired:
                launcher.kill()
        time.sleep(3)

    def listening(port):
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", int(port))) == 0

    if checks.get("banner") == "ok":
        check("summary at Ctrl-C", "tasks annotated" in text())
        check("servers stopped", not any(listening(p) for p in (PORT, ML_PORT, dashboard_port or PORT)))
    check("no traceback", "Traceback" not in text())
    print(json.dumps(checks, indent=1))
    if any(v != "ok" for v in checks.values()):
        print("--- log of the launcher ---\n" + text()[-20000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
