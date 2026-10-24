#!/usr/bin/env python3
"""
flexo_outputs.py - the GO / STOP sign servo AND the spoken advisory, in one process, both following the sentry.

    sudo python3 flexo_outputs.py                              # sign + voice, following http://localhost:8010
    sudo python3 flexo_outputs.py --voice-opts "--gain 3 --rate 170" --sign-opts "--go-angle 0 --stop-angle 180"
    sudo python3 flexo_outputs.py --no-voice                   # the sign alone
    sudo python3 flexo_outputs.py --test-voice                 # say one sentence the way the voice would, then exit
    python3 flexo_outputs.py --no-sign                         # the voice alone needs no root

It needs sign_servo.py, servo_pwm.py and tts_speaker.py in the same folder, and the sentry running. It adds nothing to
what those scripts decide: the sign follows sign_servo.py's rules (GO only after 3 NEW GO results in a row, anything else is
STOP, parked at STOP on start and exit) and the voice follows tts_speaker.py watch's (GO needs 3 new results, STOP needs 2,
a cooldown between announcements). --voice-opts and --sign-opts take the same options those scripts take.

WHY ONE PROCESS NEEDS ROOT AND THE USER AT ONCE. The servo's PWM files are root-only, but sound must play through the
logged-in user's sound session. Run under sudo, this script plays every sound AS THE USER that ran sudo (--as-user to choose),
with that user's XDG_RUNTIME_DIR, the same environment in which `python3 tts_speaker.py watch` already works by hand.

HOW IT KEEPS THE TWO APART. They run in separate threads, so a 4-second announcement never delays the sign. A voice failure
(no sound server, a player error) is logged and the sign carries on. A SIGN failure is not survivable: the sign is parked at
STOP, everything stops, and the exit code is 1, so a frozen sign can never sit at GO unnoticed.

Ctrl+C or SIGTERM: the sign is parked at STOP, then released (unless --sign-opts has --no-off-on-exit); exit code 0.
"""
import argparse
import os
import pwd
import shlex
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sign_servo as sign        # noqa: E402
import tts_speaker as tts        # noqa: E402

DEFAULT_VOICE_OPTS = "--gain 2.5 --lead-silence 0.5"
_print_lock = threading.RLock()        # reentrant: a signal handler that logs must not wait for a lock its own thread holds


def log(tag: str, msg: str) -> None:
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def make_stop_handler(got):
    """The signal handler. It runs on the main thread, between two of that thread's own steps, and the main thread spends its
    life inside stop.wait(), which briefly holds the Event's lock. A handler that called stop.set() (or printed) in that instant
    would wait for a lock its own thread holds, forever. So it only appends to a list (no lock); the main loop does the rest."""
    def request_stop(signum, frame):
        got.append(signum)
    return request_stop


def audio_identity(as_user):
    """When running as root (under sudo), the sound must be played as the logged-in user. Returns (kwargs for subprocess, user)."""
    if os.geteuid() != 0:
        return {}, None
    name = as_user or os.environ.get("SUDO_USER")
    if not name or name == "root":
        return {}, None
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        raise SystemExit(f"--as-user '{name}': no such user.")
    run_dir = f"/run/user/{pw.pw_uid}"
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": pw.pw_dir, "USER": name, "LOGNAME": name,
           "XDG_RUNTIME_DIR": run_dir, "PULSE_SERVER": f"unix:{run_dir}/pulse/native"}
    return {"user": pw.pw_uid, "group": pw.pw_gid, "extra_groups": os.getgrouplist(name, pw.pw_gid), "env": env}, pw


def run_sign(args_s, st, stop, state, heartbeat=None):
    fetch = sign.make_fetch(args_s.url, args_s.backend)
    if heartbeat is not None:
        plain = fetch

        def fetch():                                              # called once per turn of the sign loop, so a frozen loop stops reporting
            try:
                return plain()
            finally:
                heartbeat["sign"] = time.monotonic()
    try:
        sign.run(fetch, st.move_to, st.go_us, st.stop_us, st.policy, args_s.ramp, args_s.interval,
                 sleep=stop.wait, release=st.release, release_after=args_s.release_after, off_on_exit=False, stop=stop,
                 log=lambda m: log("sign", m))
    except BaseException as e:                                   # the sign must never fail silently
        state["sign_error"] = f"{type(e).__name__}: {e}"
        log("sign", f"STOPPED WITH AN ERROR: {state['sign_error']}")
        stop.set()


def run_voice(args_v, stop):
    announce = tts.make_announcer(args_v)

    def safe_announce(status, text):
        log("voice", f'{status}: "{text}"')
        try:
            announce(status, text)
        except SystemExit as e:                                   # a player or engine error: log it, keep the sign unaffected
            log("voice", f"could not speak: {e}")
        except Exception as e:                                    # noqa: BLE001
            log("voice", f"could not speak: {type(e).__name__}: {e}")

    while not stop.is_set():
        try:
            tts.watch_loop(tts.make_fetch(args_v.url, args_v.backend), lambda t: None, args_v.interval, args_v.stable, args_v.cooldown,
                           args_v.max_age_ms, sleep=stop.wait, log=lambda m: log("voice", m), announce=safe_announce,
                           fresh_only=not args_v.count_polls, stop_stable=args_v.stop_stable, stop=stop)
        except Exception as e:                                    # noqa: BLE001
            log("voice", f"loop error {type(e).__name__}: {e}; restarting in 2 s")
            stop.wait(2)


def main(argv=None, on_ready=None, heartbeat=None) -> int:
    """on_ready(): called once both outputs are running. heartbeat: a dict that receives heartbeat["sign"] = time.monotonic() on every turn of the sign loop."""
    ap = argparse.ArgumentParser(description="The GO / STOP sign servo and the voice, together, following the sentry.")
    ap.add_argument("--url", default="http://localhost:8010", help="the sentry server (default http://localhost:8010)")
    ap.add_argument("--backend", help="backend query, only if the sentry is not locked to one")
    ap.add_argument("--voice-opts", default=DEFAULT_VOICE_OPTS, help=f'options for tts_speaker.py watch (default "{DEFAULT_VOICE_OPTS}")')
    ap.add_argument("--sign-opts", default="", help="options for sign_servo.py, for example \"--go-angle 0 --stop-angle 180\"")
    ap.add_argument("--no-voice", action="store_true", help="the sign only")
    ap.add_argument("--no-sign", action="store_true", help="the voice only (no root needed)")
    ap.add_argument("--as-user", help="the user the sound plays as when running as root (default: the user who ran sudo)")
    ap.add_argument("--test-voice", action="store_true", help="say one sentence the way the voice would, then exit")
    args = ap.parse_args(argv)
    if args.no_voice and args.no_sign:
        raise SystemExit("Nothing to do: --no-voice and --no-sign together.")

    play_as, pw = audio_identity(args.as_user)
    if pw is not None:
        os.environ["HOME"] = pw.pw_dir                                   # so ~ in --voice-opts means the user's home, not root's
    tts.PLAY_AS.clear()
    tts.PLAY_AS.update(play_as)

    args_v = args_s = st = None
    if not args.no_voice:
        args_v = tts.parse_args(["watch", *shlex.split(args.voice_opts)])
        args_v.url, args_v.backend = args.url, args.backend
    if not args.no_sign:
        args_s = sign.build_parser().parse_args(shlex.split(args.sign_opts))
        args_s.url, args_s.backend = args.url, args.backend

    if args.test_voice:
        if args_v is None:
            raise SystemExit("--test-voice needs the voice (leave out --no-voice).")
        who = f" as {pw.pw_name}" if pw else ""
        log("voice", f"saying a test sentence{who}")
        tts.speak("Voice check. Everyone is wearing a vest. You may proceed.", args_v)
        return 0

    if args_s is not None:
        st = sign.setup(args_s)                                           # fails here, loudly, before anything starts
    stop = threading.Event()
    state = {}

    got = []                                                              # signal numbers, filled in by the handler
    handler = make_stop_handler(got)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    log("main", f"following {args.url}")
    if st is not None:
        log("sign", st.banner)
    if args_v is not None:
        who = f", playing as {pw.pw_name}" if pw else ""
        log("voice", f"watching: GO needs {args_v.stable} new results, STOP {args_v.stop_stable}, cooldown {args_v.cooldown:g} s{who}")

    threads = {}
    if st is not None:
        if heartbeat is not None:
            heartbeat["sign"] = time.monotonic()
        threads["sign"] = threading.Thread(target=run_sign, args=(args_s, st, stop, state, heartbeat), name="sign")
    if args_v is not None:
        threads["voice"] = threading.Thread(target=run_voice, args=(args_v, stop), name="voice", daemon=True)
    for t in threads.values():
        t.start()
    log("main", "running. Ctrl+C to stop.")
    if on_ready is not None:
        on_ready()

    while not stop.is_set() and not got:
        time.sleep(0.1)                                                   # not stop.wait(): see make_stop_handler
        if "sign" in threads and not threads["sign"].is_alive() and not stop.is_set():
            state.setdefault("sign_error", "the sign thread ended unexpectedly")
            log("main", f"the sign stopped: {state['sign_error']}")
            stop.set()

    if got:
        log("main", f"{signal.Signals(got[0]).name}: stopping")
    stop.set()
    if "sign" in threads:
        threads["sign"].join(timeout=10)                                  # its `finally` parks the sign at STOP
    if "voice" in threads:
        threads["voice"].join(timeout=6)                                  # may be finishing a sentence
    if st is not None and "sign_error" not in state and not args_s.no_off_on_exit:
        time.sleep(args_s.settle)                                         # let the servo reach STOP before it is let go
        try:
            st.release()
            log("sign", "servo off")
        except BaseException as e:                                        # noqa: BLE001
            log("sign", f"could not release the servo: {e}")
    if "sign_error" in state:
        log("main", f"exiting with an error: {state['sign_error']}")
        return 1
    log("main", "stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
