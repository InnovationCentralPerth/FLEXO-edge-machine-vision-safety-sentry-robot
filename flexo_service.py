#!/usr/bin/env python3
"""
flexo_service.py - ONE service for the whole FLEXO sentry: Ollama, the sentry web server, the GO / STOP sign and the voice.

It is what `flexo.service` runs (install_flexo_service.sh writes that unit). By hand, as root:

    sudo python3 flexo_service.py --user icp

In order, it:
  1. waits for Ollama to answer, checks that the vision model is pulled, and warns if Ollama's prompt cache is not disabled
     (LLAMA_ARG_CACHE_RAM=0, the setting that stopped the out-of-memory kills);
  2. starts the sentry web server as --user (the camera and the project belong to that user), logging into the journal as [sentry];
  3. waits until the sentry answers on /status;
  4. runs the sign and the voice (flexo_outputs.py) and tells systemd it is READY;
  5. keeps going: a sentry that dies is restarted with a back-off (the sign shows STOP meanwhile, because it only shows GO on a fresh
     GO verdict); too many crashes in a row end the service with an error so systemd restarts everything.
On SIGTERM (systemctl stop) it stops in reverse order: the sign is parked at STOP first, then the sentry.

WATCHDOG. With systemd's WatchdogSec set, it pings only while the sign loop is turning. A frozen sign loop (a stuck PWM write) stops
the pings, systemd restarts the service, and the unit's ExecStopPost parks the sign at STOP. A frozen sign sitting at GO is the one
failure that must never go unnoticed.
"""
import argparse
import json
import os
import pwd
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flexo_outputs as fo        # noqa: E402

log = fo.log

EXIT_RETRY = 1          # may fix itself (Ollama not up yet, the sentry crashed): systemd restarts the service
EXIT_CONFIG = 2         # will not fix itself (no such user, no model, no PWM chip, an old service running): the unit has RestartPreventExitStatus=2


class ConfigError(Exception):
    """A mistake in the set-up. The service exits with code 2 and systemd does not restart it."""


class Notifier:
    """sd_notify: tell systemd about readiness, status and the watchdog, through $NOTIFY_SOCKET. Does nothing outside systemd."""

    def __init__(self):
        self.addr = os.environ.get("NOTIFY_SOCKET")

    def send(self, text: str) -> None:
        if not self.addr:
            return
        addr = ("\0" + self.addr[1:]) if self.addr.startswith("@") else self.addr
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
                s.sendto(text.encode(), addr)
        except OSError as e:
            log("main", f"could not tell systemd ({e})")


def get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def ollama_cache_fix_active() -> "bool | None":
    """True if Ollama's unit sets LLAMA_ARG_CACHE_RAM=0; None if that cannot be read here."""
    try:
        out = subprocess.run(["systemctl", "show", "ollama.service", "-p", "Environment"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if not out.strip():
        return None
    return "LLAMA_ARG_CACHE_RAM=0" in out


OLD_SERVICES = ("icp-safety-sentry", "icp-safety-voice", "icp-safety-sign")


def old_services_running() -> list:
    """The older separate boot services do the same jobs and would fight this one for the port, the servo and the speaker."""
    running = []
    for name in OLD_SERVICES:
        try:
            out = subprocess.run(["systemctl", "is-active", f"{name}.service"], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return []
        if out == "active":
            running.append(name)
    return running


def wait_for_camera(path: str, wait_s: float, aborted) -> None:
    deadline = time.monotonic() + wait_s
    warned = False
    while not os.path.exists(path):
        if aborted():
            raise SystemExit(0)
        if not warned:
            log("main", f"waiting for the camera {path}")
            warned = True
        if time.monotonic() > deadline:
            raise SystemExit(f"The camera {path} is not there after {wait_s:g} s: is it plugged in?")
        time.sleep(1)


def wait_for_ollama(base: str, model: "str | None", wait_s: float, notify: Notifier, aborted) -> None:
    deadline = time.monotonic() + wait_s
    notify.send("STATUS=waiting for Ollama")
    first = True
    while True:
        if aborted():
            raise SystemExit(0)
        try:
            get_json(f"{base}/api/version")
            break
        except Exception as e:                                       # noqa: BLE001
            if first:
                log("main", f"waiting for Ollama at {base} ({type(e).__name__})")
                first = False
            if time.monotonic() > deadline:
                raise SystemExit(f"Ollama did not answer at {base} within {wait_s:g} s. Check: systemctl status ollama")
            time.sleep(1)
    log("main", "Ollama is answering")
    if model:
        try:
            names = [m.get("name", "") for m in get_json(f"{base}/api/tags").get("models", [])]
        except Exception as e:                                       # noqa: BLE001
            raise SystemExit(f"Ollama answers but its model list could not be read: {e}")
        if not any(n == model or n == f"{model}:latest" for n in names):
            raise ConfigError(f"The model '{model}' is not in Ollama (it has: {', '.join(names) or 'nothing'}). Pull it: ollama pull {model}")
        log("main", f"model {model} is present")
    fix = ollama_cache_fix_active()
    if fix is False:
        log("main", "WARNING: Ollama's prompt cache is not disabled (LLAMA_ARG_CACHE_RAM=0 is missing from its unit). On this board it grows "
                    "until the kernel kills the model process after a few minutes of continuous use. Add it to /etc/systemd/system/ollama.service.d/")


class Sentry:
    def __init__(self, args, ids):
        self.args, self.ids = args, ids
        self.proc = None
        self.lock = threading.Lock()

    def command(self):
        a = self.args
        if a.sentry_cmd:
            return shlex.split(a.sentry_cmd)
        uv = a.uv or os.path.join(self.ids["home"], ".local", "bin", "uv")
        return [uv, "run", "--no-sync", "uvicorn", "icp_safety_vlm.server.app:app", "--host", "0.0.0.0", "--port", str(a.port), "--no-access-log"]

    def start(self) -> None:
        a = self.args
        env = {"PATH": f"{self.ids['home']}/.local/bin:/usr/local/bin:/usr/bin:/bin", "HOME": self.ids["home"], "USER": self.ids["name"],
               "LOGNAME": self.ids["name"], "LOCKED_BACKEND": a.backend, "CAMERA_ROTATE": a.rotate, "PAGE_TITLE": a.page_title,
               "PYTHONUNBUFFERED": "1", "FLEXO_PORT": str(a.port)}
        for kv in a.sentry_env:
            k, _, v = kv.partition("=")
            env[k] = v
        kw = {}
        if os.geteuid() == 0 and self.ids["uid"] != 0:
            kw = {"user": self.ids["uid"], "group": self.ids["gid"], "extra_groups": self.ids["groups"]}
        proc = subprocess.Popen(self.command(), cwd=a.sentry_dir, env=env, start_new_session=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
        with self.lock:
            self.proc = proc
        threading.Thread(target=self._pump, args=(proc,), daemon=True, name="sentry-log").start()
        log("sentry", f"started (pid {proc.pid}) as {self.ids['name']}: backend {a.backend}, rotation {a.rotate}, port {a.port}")

    @staticmethod
    def _pump(proc) -> None:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log("sentry", line)

    def alive(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log("sentry", "did not stop in 10 s: killing it")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        log("sentry", "stopped")


def wait_for_sentry(sentry: Sentry, port: int, wait_s: float, notify: Notifier, aborted) -> None:
    deadline = time.monotonic() + wait_s
    notify.send("STATUS=waiting for the sentry (the model loads on the first frame)")
    t0 = time.monotonic()
    while True:
        if aborted():
            raise SystemExit(0)
        if not sentry.alive():
            raise SystemExit("The sentry stopped while starting. See the [sentry] lines above.")
        try:
            if "status" in get_json(f"http://127.0.0.1:{port}/status", timeout=2):
                log("main", f"the sentry answers after {time.monotonic() - t0:.0f} s")
                return
        except Exception:                                            # noqa: BLE001
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"The sentry did not answer on port {port} within {wait_s:g} s.")
        time.sleep(1)


def supervise(sentry: Sentry, stopping: threading.Event, fatal: dict, max_restarts: int, port: int = 0, silent_s: float = 0.0) -> None:
    """Restart a sentry that dies, or that is alive but has stopped answering. The sign and voice keep running meanwhile; with no
    fresh GO verdict the sign shows STOP."""
    recent = []
    last_answer, last_probe = time.monotonic(), 0.0
    while not stopping.wait(0.5):
        if sentry.alive():
            if silent_s > 0 and port and time.monotonic() - last_probe >= 5.0:
                last_probe = time.monotonic()
                try:
                    get_json(f"http://127.0.0.1:{port}/status", timeout=3)
                    last_answer = time.monotonic()
                except Exception:                                    # noqa: BLE001
                    if time.monotonic() - last_answer >= silent_s:
                        log("sentry", f"is running but has not answered for {time.monotonic() - last_answer:.0f} s: restarting it")
                        sentry.stop()
                        last_answer = time.monotonic()
            continue
        last_answer = time.monotonic()
        now = time.monotonic()
        recent = [t for t in recent if now - t < 120]
        code = sentry.proc.returncode if sentry.proc else "?"
        if len(recent) >= max_restarts:
            fatal["why"] = f"the sentry died {len(recent) + 1} times in 2 minutes (last exit code {code})"
            log("main", f"giving up: {fatal['why']}")
            os.kill(os.getpid(), signal.SIGTERM)                       # a clean stop of everything; the exit code will be 1
            return
        delay = min(30, 2 ** len(recent))
        recent.append(now)
        log("sentry", f"exited (code {code}); restarting in {delay} s. The sign shows STOP until it answers again.")
        if stopping.wait(delay):
            return
        try:
            sentry.start()
        except Exception as e:                                       # noqa: BLE001
            log("sentry", f"could not start: {e}")


def watchdog_loop(notify: Notifier, hb: dict, stopping: threading.Event, interval: float, max_age: float, need_sign: bool) -> None:
    warned = False
    while not stopping.wait(interval):
        age = time.monotonic() - hb.get("sign", 0.0)
        if not need_sign or age < max_age:
            notify.send("WATCHDOG=1")
            warned = False
        elif not warned:
            log("main", f"the sign loop has not turned for {age:.0f} s: NOT pinging the watchdog, so systemd will restart the service")
            warned = True


def _is_setup_problem(message: str) -> bool:
    """SystemExit messages from the sign and voice set-up that no restart will cure."""
    return any(k in message for k in ("No PWM chip", "same position", "--pitch must", "--deeper must", "must be between", "Angle must", "cannot be negative", "needs the voice"))


def identity(name: str) -> dict:
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        raise ConfigError(f"--user '{name}': no such user.")
    return {"name": name, "uid": pw.pw_uid, "gid": pw.pw_gid, "home": pw.pw_dir, "groups": os.getgrouplist(name, pw.pw_gid)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One service for the whole FLEXO sentry: Ollama, sentry, sign and voice.")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or "icp", help="the user the sentry runs as and the sound plays as (default: the sudo user, else icp)")
    ap.add_argument("--sentry-dir", help="the sentry project (default: ~/projects/icp_safety_vlm of --user)")
    ap.add_argument("--uv", help="path to uv (default: ~/.local/bin/uv of --user)")
    ap.add_argument("--backend", default="vlm_ollama_ministral")
    ap.add_argument("--rotate", default="ccw90", choices=["none", "cw90", "ccw90", "180"])
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--page-title", default="ICP FLEXO Safety Office (On-Device VLM)")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="ministral-3:3b", help="the Ollama model the backend needs (checked at start)")
    ap.add_argument("--no-model-check", action="store_true")
    ap.add_argument("--wait-ollama", type=float, default=180.0)
    ap.add_argument("--wait-sentry", type=float, default=180.0)
    ap.add_argument("--camera", default="/dev/video0", help="the camera device the sentry needs (checked before it starts)")
    ap.add_argument("--wait-camera", type=float, default=30.0)
    ap.add_argument("--no-camera-check", action="store_true")
    ap.add_argument("--sentry-silent", type=float, default=90.0, help="restart a sentry that is running but has not answered /status for this many seconds (0 = never)")
    ap.add_argument("--max-restarts", type=int, default=5, help="sentry crashes in 2 minutes before the whole service gives up")
    ap.add_argument("--gain", default="2.5")
    ap.add_argument("--go-angle", default="180")
    ap.add_argument("--stop-angle", default="0")
    ap.add_argument("--voice-opts", default="", help="more options for the voice (tts_speaker.py watch)")
    ap.add_argument("--sign-opts", default="", help="more options for the sign (sign_servo.py)")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-sign", action="store_true")
    ap.add_argument("--sentry-env", action="append", default=[], metavar="KEY=VALUE", help="an extra environment variable for the sentry (repeatable)")
    ap.add_argument("--sentry-cmd", help=argparse.SUPPRESS)                  # replaces the uv command (tests)
    ap.add_argument("--watchdog-interval", type=float, default=5.0, help=argparse.SUPPRESS)
    ap.add_argument("--watchdog-max-age", type=float, default=8.0, help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    try:
        return _main(argv)
    except ConfigError as e:
        print(f"{time.strftime('%H:%M:%S')} [main] cannot start (a set-up problem, so no automatic restart): {e}", file=sys.stderr, flush=True)
        Notifier().send(f"STATUS=set-up problem: {e}")
        return EXIT_CONFIG


def _main(argv) -> int:
    args = build_parser().parse_args(argv)
    ids = identity(args.user)
    args.sentry_dir = args.sentry_dir or os.path.join(ids["home"], "projects", "icp_safety_vlm")
    if not args.sentry_cmd and not os.path.isfile(os.path.join(args.sentry_dir, "pyproject.toml")):
        raise ConfigError(f"The sentry project is not at {args.sentry_dir} (use --sentry-dir).")
    for kv in args.sentry_env:
        if "=" not in kv or not kv.split("=", 1)[0].isidentifier():
            raise ConfigError(f"--sentry-env wants KEY=VALUE, got '{kv}'")
    if args.no_voice and args.no_sign:
        raise ConfigError("--no-voice and --no-sign together leave only the sentry: just run the sentry.")
    old = old_services_running()
    if old:
        raise ConfigError(f"the older separate service(s) {', '.join(old)} are running and do the same jobs: sudo systemctl disable --now {' '.join(old)}")
    notify = Notifier()

    early = []                                                         # a stop request that arrives while we are still starting up
    early_handler = fo.make_stop_handler(early)
    signal.signal(signal.SIGINT, early_handler)
    signal.signal(signal.SIGTERM, early_handler)
    aborted = lambda: bool(early)

    sentry = Sentry(args, ids)
    stopping = threading.Event()
    fatal, hb = {}, {}
    try:
        wait_for_ollama(args.ollama_url, None if args.no_model_check else args.model, args.wait_ollama, notify, aborted)
        if not args.no_camera_check:
            wait_for_camera(args.camera, args.wait_camera, aborted)
        sentry.start()
        wait_for_sentry(sentry, args.port, args.wait_sentry, notify, aborted)
        threading.Thread(target=supervise, args=(sentry, stopping, fatal, args.max_restarts, args.port, args.sentry_silent), daemon=True,
                         name="sentry-supervisor").start()

        wd_usec = os.environ.get("WATCHDOG_USEC")
        interval = min(args.watchdog_interval, float(wd_usec) / 2e6) if wd_usec else args.watchdog_interval

        def on_ready():
            notify.send("READY=1\nSTATUS=running: Ollama, the sentry"
                        + ("" if args.no_sign else ", the sign") + ("" if args.no_voice else ", the voice"))
            threading.Thread(target=watchdog_loop, args=(notify, hb, stopping, interval, args.watchdog_max_age, not args.no_sign),
                             daemon=True, name="watchdog").start()

        out_argv = ["--url", f"http://127.0.0.1:{args.port}", "--as-user", args.user,
                    "--voice-opts", f"--gain {args.gain} --lead-silence 0.5 {args.voice_opts}".strip(),
                    "--sign-opts", f"--go-angle {args.go_angle} --stop-angle {args.stop_angle} --no-off-on-exit {args.sign_opts}".strip()]
        if args.no_voice:
            out_argv.append("--no-voice")
        if args.no_sign:
            out_argv.append("--no-sign")
        if early:                                                       # asked to stop during start-up
            return 0
        rc = fo.main(out_argv, on_ready=on_ready, heartbeat=hb)
    except ConfigError:
        stopping.set()
        sentry.stop()
        raise
    except SystemExit as e:
        stopping.set()
        notify.send("STOPPING=1")
        sentry.stop()
        if e.code in (None, 0):
            return 0
        log("main", f"cannot start: {e.code}")
        notify.send(f"STATUS=failed: {e.code}")
        return EXIT_CONFIG if _is_setup_problem(str(e.code)) else EXIT_RETRY
    except BaseException as e:                                          # noqa: BLE001
        stopping.set()
        sentry.stop()
        log("main", f"unexpected error: {type(e).__name__}: {e}")
        return EXIT_RETRY
    stopping.set()
    notify.send("STOPPING=1")
    sentry.stop()
    if fatal:
        return EXIT_RETRY
    return rc


if __name__ == "__main__":
    sys.exit(main())
