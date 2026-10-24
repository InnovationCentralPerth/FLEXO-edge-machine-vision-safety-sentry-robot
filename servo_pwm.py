#!/usr/bin/env python3
"""
servo_pwm.py - drive a hobby servo from a Jetson Orin Nano's hardware PWM.

Talks to the kernel's sysfs PWM interface directly (no Jetson.GPIO, no
dependencies). Default pin is header pin 33, which jetson-io calls "pwm5".

ONE-TIME SETUP
  1. Route the PWM controller to the pin, then reboot:
         sudo /opt/nvidia/jetson-io/jetson-io.py
           -> Configure Jetson 40pin Header
           -> Configure header pins manually
           -> tick  pwm5 (33)
           -> Back -> Save pin changes -> Save and reboot
  2. After the reboot, run `probe` (below) to see which frame lengths this
     PWM block will actually enable.

WIRING (3.3 V logic)
  servo signal  -> header pin 33
  servo ground  -> any header GND (pin 34 is next to it) AND the servo
                   supply's ground. The grounds must be joined.
  servo power   -> a separate 5-6 V supply sized for the servo, not the
                   header. A stalled servo can draw far more than a header
                   pin should supply and can brown out the board.
  A 470-1000 uF capacitor across the servo supply helps with twitching.

COMMANDS (run with sudo; sysfs PWM files are root-only by default)
  sudo python3 servo_pwm.py angle 90         move to 90 degrees
  sudo python3 servo_pwm.py angle 0 --ramp 0.6    move over 0.6 s
  sudo python3 servo_pwm.py pulse 1500       set the pulse width in microseconds
  sudo python3 servo_pwm.py sweep            slow sweep across the safe range
  sudo python3 servo_pwm.py off              stop the signal (servo goes limp)
  sudo python3 servo_pwm.py info             show the channel's current period / duty / enable
  sudo python3 servo_pwm.py probe            find which periods this PWM block will actually enable
  sudo python3 servo_pwm.py --period-ms 10 angle 90     use a different servo frame length

The PWM keeps running after this script exits, which is what holds the servo
in position. Run "off" to release it.

RANGE: the default pulse range is 1000-2000 us for 0-180 degrees, which is
conservative, and on most servos it turns only about half of the full travel.
Many 180-degree servos reach their full travel at 500-2500 us: add --wide for
that, or set your own with --min-us / --max-us. Do this with a short test and
listen: if the servo hums or strains at an end, back off 50-100 us, because
driving past its end stops stalls it and can burn it out. A hard limit of
400-2600 us is enforced.

360 DEGREES: a normal (positional) servo CANNOT turn 360 degrees. It has a
mechanical stop and a position sensor that covers about 180 (some 270)
degrees, and no pulse width changes that. Full turns need a continuous-rotation
servo, where the pulse sets speed and direction, not angle:
  sudo python3 servo_pwm.py spin 0.5 --seconds 2       speed -1..1 for 2 s, then stop
  sudo python3 servo_pwm.py turn 360 --sec-per-rev 2.8  one turn, timed (no feedback)
Use these ONLY with a continuous-rotation servo. They are open loop: the angle
is only as good as your --sec-per-rev measurement and it drifts. The neutral
(stop) pulse differs from servo to servo; trim it with --neutral-us until
`spin 0` is completely still.
"""

import argparse
import errno
import glob
import math
import os
import sys
import time

# Pin -> PWM controller device name, from the Orin Nano 40-pin header table.
PIN_CHIP = {33: "32c0000.pwm", 15: "3280000.pwm"}
PERIOD_NS = 20_000_000          # 50 Hz, standard servo frame (change with --period-ms)
# Periods tried by `probe`, longest first. Each is enabled with a centred 1.5 ms pulse,
# which is harmless to a connected servo.
PROBE_PERIODS_NS = (1_000_000_000, 100_000_000, 50_000_000, 20_000_000, 15_000_000,
                    10_000_000, 8_000_000, 6_000_000, 5_000_000, 4_000_000, 3_000_000)
HARD_MIN_US, HARD_MAX_US = 400, 2600


def find_chip(match: str, root: str = "/sys/class/pwm") -> str:
    chips = sorted(glob.glob(os.path.join(root, "pwmchip*")))
    for c in chips:
        if match in os.path.realpath(c):
            return c
    listing = "\n".join(f"    {c} -> {os.path.realpath(c)}" for c in chips) or "    (none)"
    raise SystemExit(
        f"No PWM chip matching '{match}' under {root}.\nChips present:\n{listing}\n"
        "If the list is empty or has no match, the PWM controller is not enabled in this "
        "kernel/device tree. Pass --chip-match with a substring from the list to try another."
    )


def _write(path: str, value) -> None:
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except PermissionError:
        raise SystemExit(f"Permission denied writing {path}. Run with sudo.")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


class Servo:
    def __init__(self, chip_dir: str, channel: int = 0):
        self.chip = chip_dir
        self.dir = os.path.join(chip_dir, f"pwm{channel}")
        self.channel = channel

    def export(self) -> None:
        if not os.path.isdir(self.dir):
            try:
                _write(os.path.join(self.chip, "export"), self.channel)
            except OSError as e:
                if e.errno == errno.EBUSY:
                    raise SystemExit(f"{self.chip} channel {self.channel} is in use by another driver "
                                     "(the fan, for example). Pick a different PWM controller.")
                raise
            for _ in range(40):                       # up to 2 s for the node to appear
                if os.path.isdir(self.dir):
                    break
                time.sleep(0.05)
            else:
                raise SystemExit(f"{self.dir} did not appear after export.")

    def _p(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _release_output(self) -> None:
        if _read(self._p("enable")) == "1":
            _write(self._p("enable"), 0)                  # period can only change while disabled on some drivers

    def _configure(self, period_ns: int, duty_ns: int) -> None:
        # The kernel keeps the last period and duty after a channel is released. It
        # rejects a new period shorter than the duty it still holds (a 20 ms servo
        # period after an earlier run used 1 s / 0.5 s, for example), so lower the duty
        # first. Writing duty 0 on a fresh channel (period 0) would be rejected
        # too, so only do it when a stale, larger duty is actually there.
        if int(_read(self._p("duty_cycle"))) > period_ns:
            _write(self._p("duty_cycle"), 0)
        _write(self._p("period"), period_ns)
        _write(self._p("duty_cycle"), duty_ns)

    def start(self, period_ns: int, duty_ns: int, explain: bool = True) -> None:
        """explain=False re-raises the kernel's OSError instead of probing and explaining."""
        self.export()
        self._release_output()
        try:
            self._configure(period_ns, duty_ns)
        except OSError as e:
            if e.errno != errno.EINVAL or not explain:
                raise
            raise SystemExit(self._explain_rejected_period(period_ns))
        try:
            _write(self._p("enable"), 1)
        except OSError as e:
            if e.errno != errno.EINVAL or not explain:
                raise
            raise SystemExit(self._explain_rejected_enable(period_ns))

    def state(self) -> dict:
        if not os.path.isdir(self.dir):
            return {}
        return {k: _read(self._p(k)) for k in ("period", "duty_cycle", "enable")}

    def probe(self, periods_ns=PROBE_PERIODS_NS):
        """Try to ENABLE the channel at each period, with a centred 1.5 ms pulse. Leaves it off."""
        self.export()
        results = []
        for p in periods_ns:
            self._release_output()
            self._configure(p, 1_500_000)
            try:
                _write(self._p("enable"), 1)
                ok = True
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
                ok = False
            results.append((p, ok))
            self._release_output()
        return results

    def _explain_rejected_enable(self, period_ns: int) -> str:
        """Period and duty were accepted, but turning the output on was refused."""
        lines = [f"The kernel accepted a period of {period_ns} ns ({1e9 / period_ns:.0f} Hz) but would not "
                 "enable the output (EINVAL).", "Testing which periods it will enable:\n"]
        lines += describe_probe(self.probe(), period_ns)
        lines.append("\nThe PWM is left off. Send me this whole message if you want a second opinion.")
        return "\n".join(lines)

    def _explain_rejected_period(self, period_ns: int) -> str:
        """The kernel refused to take the period at all, even after the stale-duty fix."""
        before = self.state()
        accepted = None
        for p in (10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000, 100_000):
            try:
                _write(self._p("period"), p)
                accepted = p
                break
            except OSError:
                continue
        lines = [f"The kernel rejected a period of {period_ns} ns ({1e9 / period_ns:.0f} Hz).",
                 f"Channel state before: {before}"]
        if accepted:
            lines.append(f"The longest period it accepted was {accepted / 1e6:g} ms. This PWM block may not be able "
                         "to produce a 20 ms servo frame (try --period-ms, see the servo's datasheet first).")
        else:
            lines.append("It rejected every period tried, down to 0.1 ms. The channel may not be usable: "
                         "check the pin routing (jetson-io) and `sudo cat /sys/kernel/debug/pwm`.")
        lines.append("Reset with `sudo python3 servo_pwm.py off`, then send me this message.")
        return "\n".join(lines)

    def duty_ns(self) -> int:
        return int(_read(os.path.join(self.dir, "duty_cycle")))

    def set_duty_ns(self, duty_ns: int) -> None:
        _write(os.path.join(self.dir, "duty_cycle"), duty_ns)

    def is_enabled(self) -> bool:
        return os.path.isdir(self.dir) and _read(os.path.join(self.dir, "enable")) == "1"

    def off(self) -> None:
        if os.path.isdir(self.dir):
            _write(os.path.join(self.dir, "enable"), 0)
            _write(os.path.join(self.chip, "unexport"), self.channel)


def set_period_ms(ms: float) -> None:
    """Use a servo frame other than 20 ms, for example when the PWM block cannot make 50 Hz."""
    global PERIOD_NS
    if not (3.0 <= ms <= 50.0):
        raise SystemExit("--period-ms must be between 3 and 50 (a servo pulse can be up to 2.6 ms long).")
    PERIOD_NS = int(round(ms * 1_000_000))


def describe_probe(results, wanted_ns: int):
    """Plain-language reading of a probe: [(period_ns, enabled_ok), ...]."""
    lines = ["  period       rate    can enable"]
    for p, ok in results:
        lines.append(f"  {p / 1e6:7g} ms  {1e9 / p:6.1f} Hz   {'yes' if ok else 'REJECTED'}")
    in_range = [p for p, ok in results if ok and 3_000_000 <= p <= 25_000_000]
    longer = [p for p, ok in results if ok and p > 25_000_000]
    if wanted_ns in [p for p, ok in results if ok]:
        lines.append(f"\n{wanted_ns / 1e6:g} ms DID enable this time, so the earlier failure was not a period limit. "
                     "Run the command again.")
    elif in_range:
        best = max(in_range)
        lines += [f"\nThe longest servo-range period this PWM block will enable is {best / 1e6:g} ms "
                  f"({1e9 / best:.0f} Hz). The request that failed was {wanted_ns / 1e6:g} ms.",
                  f"Try:  sudo python3 servo_pwm.py --period-ms {best / 1e6:g} angle 90",
                  "A servo expects about 50 Hz. Many digital servos accept 100-300 Hz, but some analog ones "
                  "jitter or overheat when refreshed faster. Check the datasheet and keep tests short. "
                  "Another PWM controller on the chip may have a different limit (`ls -l /sys/class/pwm/`; "
                  "probe each with --chip-match). A PCA9685 servo driver board avoids the question."]
    elif longer:
        lines += ["\nIt enables long periods but nothing in the 3-25 ms servo range. This PWM block cannot make "
                  "servo-rate signals as configured. A PCA9685 servo driver board (I2C) is the clean fix."]
    else:
        lines += ["\nNothing could be enabled. The channel may not be routed or usable: check the pin routing "
                  "(jetson-io, then reboot), `sudo dmesg | tail -20` and `sudo cat /sys/kernel/debug/pwm`."]
    return lines


def clamp_pulse(us: float) -> int:
    if not (HARD_MIN_US <= us <= HARD_MAX_US):
        raise SystemExit(f"Pulse {us:.0f} us is outside the hard limit {HARD_MIN_US}-{HARD_MAX_US} us.")
    return int(round(us))


def angle_to_us(angle: float, min_us: int, max_us: int) -> int:
    if not (0 <= angle <= 180):
        raise SystemExit("Angle must be between 0 and 180.")
    return clamp_pulse(min_us + (max_us - min_us) * angle / 180.0)


def move(servo: Servo, target_us: int, ramp_s: float) -> None:
    """Move to target_us. If the signal is already running, ramp from the current pulse."""
    if not servo.is_enabled():
        servo.start(PERIOD_NS, target_us * 1000)   # first move is immediate: the servo's start position is unknown
        return
    start_us = servo.duty_ns() / 1000.0
    steps = max(int(ramp_s / 0.02), 1)
    for i in range(1, steps + 1):
        servo.set_duty_ns(int((start_us + (target_us - start_us) * i / steps) * 1000))
        if ramp_s > 0:
            time.sleep(ramp_s / steps)


def spin(servo, speed: float, seconds: float, neutral_us: int, span_us: int, ramp_s: float) -> int:
    """Run a CONTINUOUS-ROTATION servo at `speed` (-1..1) for `seconds`, then stop.
    Returns the pulse used. The signal is left running at the stop pulse."""
    if not (-1.0 <= speed <= 1.0):
        raise SystemExit("Speed must be between -1 and 1.")
    neutral = clamp_pulse(neutral_us)
    target = clamp_pulse(neutral_us + speed * span_us)
    move(servo, neutral, 0)                       # start (or stay) at the stop pulse
    if speed != 0 and seconds > 0:
        move(servo, target, ramp_s)
        time.sleep(seconds)
        move(servo, neutral, ramp_s)              # back to stopped
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Hobby-servo control on Jetson hardware PWM.")
    ap.add_argument("command", choices=["angle", "pulse", "sweep", "spin", "turn", "off", "info", "probe"])
    ap.add_argument("value", nargs="?", type=float, help="degrees for 'angle' and 'turn', microseconds for 'pulse', speed -1..1 for 'spin'")
    ap.add_argument("--pin", type=int, default=33, choices=sorted(PIN_CHIP))
    ap.add_argument("--chip-match", help="substring of the PWM chip's device path (overrides --pin)")
    ap.add_argument("--root", default="/sys/class/pwm", help=argparse.SUPPRESS)
    ap.add_argument("--period-ms", type=float, default=20.0,
                    help="servo frame length; only change it if `probe` says the PWM cannot make 20 ms")
    ap.add_argument("--wide", action="store_true",
                    help="full-travel preset for most 180-degree servos: --min-us 500 --max-us 2500")
    ap.add_argument("--min-us", type=int, default=None, help="pulse for 0 degrees (default 1000, or 500 with --wide)")
    ap.add_argument("--max-us", type=int, default=None, help="pulse for 180 degrees (default 2000, or 2500 with --wide)")
    ap.add_argument("--ramp", type=float, default=0.0, help="seconds to take over a move")
    ap.add_argument("--neutral-us", type=int, default=1500, help="spin/turn: the pulse at which a continuous servo is still")
    ap.add_argument("--span-us", type=int, default=400, help="spin/turn: pulse change from neutral at full speed")
    ap.add_argument("--seconds", type=float, default=1.0, help="spin: how long to run")
    ap.add_argument("--speed", type=float, default=0.5, help="turn: speed 0..1 to use")
    ap.add_argument("--sec-per-rev", type=float, help="turn: seconds one full revolution takes at --speed (measure it)")
    args = ap.parse_args(argv)

    set_period_ms(args.period_ms)
    default_range = args.min_us is None and args.max_us is None and not args.wide
    if args.min_us is None:
        args.min_us = 500 if args.wide else 1000
    if args.max_us is None:
        args.max_us = 2500 if args.wide else 2000
    if args.min_us >= args.max_us:
        raise SystemExit("--min-us must be smaller than --max-us.")
    servo = Servo(find_chip(args.chip_match or PIN_CHIP[args.pin], args.root))

    if args.command == "probe":
        print("Trying to enable the PWM at several periods, each with a centred 1.5 ms pulse "
              "(harmless to a connected servo)...\n")
        results = servo.probe()
        servo.off()
        print("\n".join(describe_probe(results, PERIOD_NS)))
    elif args.command in ("angle", "pulse"):
        if args.value is None:
            raise SystemExit(f"'{args.command}' needs a value.")
        us = angle_to_us(args.value, args.min_us, args.max_us) if args.command == "angle" \
            else clamp_pulse(args.value)
        move(servo, us, args.ramp)
        print(f"Pulse {us} us. The signal keeps running; use 'off' to release the servo.")
        if args.command == "angle" and default_range:
            print("Note: 1000-2000 us is mapped to 0-180 degrees here, but most 180-degree servos need "
                  "about 500-2500 us for their full travel. If it turns less than you asked, add --wide.")
    elif args.command == "sweep":
        move(servo, args.min_us, 0)
        time.sleep(1)
        for target in (args.max_us, args.min_us, (args.min_us + args.max_us) // 2):
            move(servo, clamp_pulse(target), 2.0)
            time.sleep(0.5)
        print("Sweep done, parked at centre.")
    elif args.command in ("spin", "turn"):
        print("For CONTINUOUS-ROTATION servos only. A normal servo cannot turn past its stops; "
              "spinning one just drives it to the end of its travel.")
        if args.value is None:
            raise SystemExit(f"'{args.command}' needs a value "
                             f"({'speed -1..1' if args.command == 'spin' else 'degrees, e.g. 360 or -180'}).")
        if args.command == "spin":
            us = spin(servo, args.value, args.seconds, args.neutral_us, args.span_us, args.ramp)
            print(f"Ran at pulse {us} us for {args.seconds:g} s, now stopped at {args.neutral_us} us. "
                  "If it creeps, adjust --neutral-us until `spin 0` is still. Use 'off' to release.")
        else:
            if not args.sec_per_rev or args.sec_per_rev <= 0:
                raise SystemExit("'turn' needs --sec-per-rev: the seconds one full revolution takes at the speed "
                                 "you use. Measure it with:  spin 0.5 --seconds 5  and count the turns "
                                 "(seconds per turn = 5 / turns).")
            if not (0 < args.speed <= 1):
                raise SystemExit("--speed must be above 0 and at most 1.")
            seconds = abs(args.value) / 360.0 * args.sec_per_rev
            us = spin(servo, math.copysign(args.speed, args.value), seconds, args.neutral_us,
                      args.span_us, args.ramp)
            print(f"Turned {args.value:g} degrees by running {seconds:.2f} s at pulse {us} us. "
                  "This is open loop (no angle feedback), so it drifts. Use 'off' to release.")
    elif args.command == "info":
        st = servo.state()
        print(f"PWM chip: {servo.chip} -> {os.path.realpath(servo.chip)}")
        if st:
            print(f"channel 0: period={st['period']} ns  duty_cycle={st['duty_cycle']} ns  enable={st['enable']}")
        else:
            print("channel 0 is not exported (nothing running).")
    else:
        servo.off()
        print("PWM off.")


if __name__ == "__main__":
    main()
