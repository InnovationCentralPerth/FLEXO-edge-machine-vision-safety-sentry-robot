#!/usr/bin/env python3
"""
sign_servo.py - flip a GO / STOP sign with a servo, following the sentry.

Polls the sentry server's /status and moves the servo through servo_pwm.py
(keep both files in the same folder).

POLICY: FAIL-SAFE. The sign shows GO only when a vest is confirmed. Everything
else shows STOP: STOP, NO PERSON, STARTING, ERROR, a stale verdict, and a
server that cannot be reached.
  - GO must hold for --stable NEW results in a row before the sign moves to it. The sentry
    makes a result about every 2 seconds but this script polls every 0.3 s, so the same
    result is seen several times; only a result whose age is no larger than at the
    previous poll is new. One stray GO result can therefore never move the sign. A server
    that reports no age counts every poll. --count-polls restores the old behaviour.
  - Leaving GO is immediate: the first poll that is not a fresh GO flips it.
  - A verdict older than --max-age-ms counts as not GO.
  - If the server cannot be reached for --timeout seconds, the sign goes STOP.
  - The sign is parked at STOP when this script starts and when it exits.
  To keep the last position while nobody is in frame, change the NO PERSON
  handling in Policy.update().

ANGLES. By default STOP is angle 0 and GO is angle 180, the same angles that
`servo_pwm.py --period-ms 5 --wide angle 0` and `angle 180` move to (500 and 2500 us).
If your sign shows them the other way round, swap them:  --go-angle 0 --stop-angle 180
(--min-us and --max-us set the pulses for 0 and 180 degrees; --go-us / --stop-us give a
pulse directly and win over the angles). The servo is HELD while the script runs.
OFF. When the script ends (Ctrl+C or SIGTERM) it parks at STOP, waits --settle seconds for
the servo to get there, then releases it like `servo_pwm.py off` (--no-off-on-exit keeps
holding STOP). --release-after N releases the servo N seconds after each move, for a sign
that does not need holding torque; the next change drives it again.

CALIBRATE FIRST. Find the pulse widths where the sign actually shows GO and
STOP, with the sign mechanism attached and a servo supply that is ready:
    sudo python3 servo_pwm.py pulse 1000        # look at the sign, adjust
    sudo python3 servo_pwm.py pulse 2000
Many servos need about 500-2500 us for a full 180-degree turn. Then pass them:
    sudo python3 sign_servo.py --go-us 600 --stop-us 2400
The defaults (1000 and 2000) are conservative and may give only part of a turn.
Pulses outside 400-2600 us are refused.

TRY IT WITHOUT A SERVO:
    python3 sign_servo.py --dry-run          prints GO / STOP decisions only

IF THIS SCRIPT IS KILLED, the PWM keeps running at its last position, which
could be GO. When running it under systemd, park the sign on every stop or
crash:
    [Service]
    ExecStart=/usr/bin/python3 /home/icp/servo/sign_servo.py --go-us 600 --stop-us 2400
    ExecStopPost=/usr/bin/python3 /home/icp/servo/servo_pwm.py pulse 2400
    Restart=on-failure
(The sysfs PWM files are root-only by default, so run the unit as root or add
a udev rule.) A power cut to the Jetson leaves the servo unpowered, so a
mechanical default (for example a spring that falls to STOP) is worth having.
"""

import argparse
import json
import signal
import sys
import time
import types
import urllib.parse
import urllib.request

from servo_pwm import PIN_CHIP, Servo, angle_to_us, clamp_pulse, find_chip, move, set_period_ms


class Policy:
    """Decides GO or STOP from each poll. data=None means the poll failed."""

    def __init__(self, stable=3, max_age_ms=5000, timeout_s=3.0, fresh_only=True):
        self.stable, self.max_age_ms, self.timeout_s, self.fresh_only = stable, max_age_ms, timeout_s, fresh_only
        self.state = "STOP"
        self.go_count = 0
        self.last_ok = None
        self.prev_age = None

    def _stop(self):
        self.go_count = 0
        self.state = "STOP"

    def update(self, data, now):
        if data is None:                                   # server unreachable
            self.prev_age = None
            if self.last_ok is None or now - self.last_ok >= self.timeout_s:
                self._stop()
            return self.state
        self.last_ok = now
        age = data.get("age_ms")
        fresh = (age if age is not None else 0) <= self.max_age_ms          # not stale
        # a NEW result: its age has not grown since the last poll (a sentry that sends no age counts every poll)
        new_result = (not self.fresh_only) or age is None or self.prev_age is None or age <= self.prev_age
        self.prev_age = age
        if data.get("status") == "GO" and fresh:
            if new_result:
                self.go_count += 1
            if self.go_count >= self.stable:
                self.state = "GO"
        else:                                              # anything else, including NO PERSON, closes
            self._stop()
        return self.state


def run(fetch, move_to, go_us, stop_us, policy, ramp=0.4, interval=0.3,
        sleep=time.sleep, now=time.monotonic, max_polls=None, log=print,
        release=None, release_after=0.0, off_on_exit=False, settle=0.8, stop=None):
    """Poll, decide, and move the sign only when the decision changes.
    release: a function that switches the servo off (like `servo_pwm.py off`). With release_after > 0 it is
    called that many seconds after the last move, and the next move drives the servo again. With off_on_exit it
    is called once the sign has been parked at STOP for good, after `settle` seconds for the servo to arrive.
    stop: an optional threading.Event; the loop ends when it is set, and the sign is parked at STOP as always."""
    move_to(stop_us, 0.0)                                  # known safe start
    current = "STOP"
    log("Sign parked at STOP.")
    last_move, released = now(), False
    polls = 0
    try:
        while (max_polls is None or polls < max_polls) and not (stop is not None and stop.is_set()):
            polls += 1
            try:
                data = fetch()
            except KeyboardInterrupt:
                raise
            except Exception:
                data = None
            want = policy.update(data, now())
            if want != current:
                shown = (data or {}).get("status", "no response")
                log(f"{time.strftime('%H:%M:%S')}  {want}   (sentry says: {shown})")
                move_to(go_us if want == "GO" else stop_us, ramp)
                current, last_move, released = want, now(), False
            elif release and release_after > 0 and not released and now() - last_move >= release_after:
                release()                                  # no holding torque needed: stop driving the servo
                released = True
                log(f"{time.strftime('%H:%M:%S')}  servo released after {release_after:g} s at {current}")
            sleep(interval)
    finally:
        move_to(stop_us, 0.0)                              # always leave the sign at STOP
        log("Sign parked at STOP.")
        if release and off_on_exit:
            sleep(settle)                                  # let the servo get there before it is let go
            release()
            log("Servo off.")


def make_fetch(url, backend):
    full = url.rstrip("/") + "/status"
    if backend:
        full += "?" + urllib.parse.urlencode({"backend": backend})

    def fetch():
        with urllib.request.urlopen(full, timeout=2) as r:
            return json.load(r)
    return fetch


def _terminate(*_):
    raise KeyboardInterrupt                                # lets `finally` park the sign on SIGTERM


def build_parser():
    ap = argparse.ArgumentParser(description="GO / STOP sign servo, driven by the sentry.")
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--backend", help="backend query (omit if the server is locked)")
    ap.add_argument("--pin", type=int, default=33, choices=sorted(PIN_CHIP))
    ap.add_argument("--chip-match")
    ap.add_argument("--root", default="/sys/class/pwm", help=argparse.SUPPRESS)
    ap.add_argument("--period-ms", type=float, default=5.0,
                    help="servo frame length in ms (default 5: this board's PWM only enables 5 ms or shorter; see servo_pwm.py probe)")
    ap.add_argument("--go-angle", type=float, default=180.0, help="servo angle that shows GO, 0-180 (default 180)")
    ap.add_argument("--stop-angle", type=float, default=0.0, help="servo angle that shows STOP, 0-180 (default 0)")
    ap.add_argument("--min-us", type=int, default=500, help="pulse for 0 degrees (default 500, the --wide range)")
    ap.add_argument("--max-us", type=int, default=2500, help="pulse for 180 degrees (default 2500, the --wide range)")
    ap.add_argument("--go-us", type=int, help="servo pulse that shows GO (instead of --go-angle)")
    ap.add_argument("--stop-us", type=int, help="servo pulse that shows STOP (instead of --stop-angle)")
    ap.add_argument("--ramp", type=float, default=0.4, help="seconds a flip takes")
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--stable", type=int, default=3, help="new GO results in a row before showing GO (default 3)")
    ap.add_argument("--count-polls", action="store_true", help="count every poll, not only new results (the old behaviour)")
    ap.add_argument("--max-age-ms", type=int, default=5000)
    ap.add_argument("--timeout", type=float, default=3.0, help="seconds without the server before STOP")
    ap.add_argument("--release-after", type=float, default=0.0, help="release the servo this many seconds after each move (default 0: keep holding)")
    ap.add_argument("--no-off-on-exit", action="store_true", help="keep holding STOP when the script ends instead of releasing the servo")
    ap.add_argument("--settle", type=float, default=0.8, help="seconds to let the servo reach STOP before releasing it at exit")
    ap.add_argument("--dry-run", action="store_true", help="print decisions, do not touch the servo")
    return ap


def setup(args):
    """Everything main() prepares before polling: the two pulses, the policy, and the functions that move and release the
    servo. Returns a namespace so another script (flexo_outputs.py) can run the same loop. Raises SystemExit on bad values."""
    set_period_ms(args.period_ms)
    go_us = clamp_pulse(args.go_us) if args.go_us is not None else angle_to_us(args.go_angle, args.min_us, args.max_us)
    stop_us = clamp_pulse(args.stop_us) if args.stop_us is not None else angle_to_us(args.stop_angle, args.min_us, args.max_us)
    if go_us == stop_us:
        raise SystemExit("GO and STOP are the same position. Give --go-angle and --stop-angle different values.")
    if args.release_after < 0 or args.settle < 0:
        raise SystemExit("--release-after and --settle cannot be negative.")
    policy = Policy(args.stable, args.max_age_ms, args.timeout, fresh_only=not args.count_polls)

    if args.dry_run:
        def move_to(us, ramp):
            print(f"   [dry run] servo -> {us} us")

        def release():
            print("   [dry run] servo off")
    else:
        servo = Servo(find_chip(args.chip_match or PIN_CHIP[args.pin], args.root))
        move_to = lambda us, ramp: move(servo, us, ramp)
        release = servo.off
    go_txt = f"{go_us} us" + (f" ({args.go_angle:g} deg)" if args.go_us is None else "")
    stop_txt = f"{stop_us} us" + (f" ({args.stop_angle:g} deg)" if args.stop_us is None else "")
    banner = f"GO={go_txt}, STOP={stop_txt}, frame {args.period_ms:g} ms"
    return types.SimpleNamespace(go_us=go_us, stop_us=stop_us, policy=policy, move_to=move_to, release=release, banner=banner)


def main(argv=None):
    args = build_parser().parse_args(argv)
    st = setup(args)
    signal.signal(signal.SIGTERM, _terminate)
    print(f"Following {args.url}. {st.banner}. Ctrl+C to stop.")
    try:
        run(make_fetch(args.url, args.backend), st.move_to, st.go_us, st.stop_us, st.policy,
            args.ramp, args.interval, release=st.release, release_after=args.release_after,
            off_on_exit=not args.no_off_on_exit, settle=args.settle)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
