#!/usr/bin/env python3
"""
tts_speaker.py - offline text-to-speech out of a speaker on the Jetson.

Two engines, both fully offline:
  piper     neural voice, much more natural. Needs `pip install piper-tts` and a
            downloaded voice file.
  espeak    espeak-ng, robotic but tiny and works with one apt package.
"auto" (the default) uses Piper when it and the voice file are present, and
falls back to espeak-ng otherwise.

AUDIO OUTPUT
  Nano-family dev kits have had no 3.5 mm audio jack (NVIDIA confirmed this for
  the original Nano; check your Orin Nano with `aplay -l`). The easy options:
    - a USB speaker or USB sound card (plug and play)
    - the speakers of a DisplayPort / HDMI monitor (shows up as "tegra-hda")
  Find the card and device numbers:    python3 tts_speaker.py devices
  Then pass them:                      --device plughw:2,0     (card 2, device 0)
  Without --device the system default output is used.

SETUP
  sudo apt install -y alsa-utils espeak-ng
  python3 -m venv ~/tts-env && source ~/tts-env/bin/activate
  pip install piper-tts                  # optional; espeak-ng works without it
  mkdir -p ~/piper-voices && cd ~/piper-voices
  python3 -m piper.download_voices en_US-lessac-medium
  (Piper is GPL-licensed and each voice has its own licence; check both before
   you redistribute anything.)

LOUDNESS
  Three places, in order of effect:
    1. the monitor / TV volume (HDMI volume is set there, not on the Jetson)
    2. this script: every sentence is scaled up to near full level, and
       --gain 2 (or 3) boosts it further with a soft limiter. Above about 3
       it starts to sound distorted.
    3. a louder speaker: an HDMI audio extractor feeding powered speakers
  --no-normalize turns the automatic scaling off.

PLAYERS (what makes the sound)
  No --device: the sound goes through the desktop's sound server (paplay, then pw-play, then aplay). On a Jetson
  with the desktop running this is the one that works; the desktop holds the HDMI hardware.
  --device plughw:0,3: straight to that ALSA device with aplay. Use it on a headless board, or with the desktop
  stopped; with the desktop running it fails with "Device or resource busy".
  --player aplay|paplay|pw-play forces one, which is the quick way to find the route that reaches your speaker.

WATCH (read the sentry's verdict aloud)
  watch --url http://localhost:8010 --backend vlm_ollama_ministral --device plughw:0,3 --gain 2.5
  It speaks the sentry's own message when the verdict changes (or the stock line, or your clip). A verdict must
  hold across NEW VLM results, not just repeated polls: --stable 3 for GO and --stop-stable 2 for STOP, so one
  stray result is never announced. --cooldown 6 keeps announcements apart, NO PERSON re-arms it, and a result
  older than --max-age-ms is ignored. --count-polls restores the old poll counting.

RECORDED CLIPS (your own recordings instead of a synthetic voice)
  --clips DIR        a folder holding go.wav and stop.wav. In `watch`, a GO or STOP verdict plays that
                     recording instead of synthesising the sentence. A missing clip falls back to the
                     synthetic voice. `clip NAME --clips DIR` plays one clip by hand (clip stop, clip go).
  Loudness (--gain, --no-normalize) and --lead-silence still apply. Voice effects and --deeper never touch a
  recording: the recording is the voice. Use 16-bit WAV, mono; convert others with
  ffmpeg -i in.m4a -ac 1 -ar 48000 -c:a pcm_s16le go.wav

VOICE STYLE (all optional, off by default; none of it imitates any particular character)
  --pitch 30         espeak-ng only: 0-99, lower is deeper (default 50)
  --deeper 3         any engine: lower the voice by 3 semitones (also makes it a little slower)
  --effect metal     a ringing robot timbre; --effect lofi gives a crunchy radio sound
  e.g.  say "..." --engine espeak --espeak-voice "en-us+m3+croak" --rate 130 --pitch 30 --effect metal
  espeak-ng voice variants such as m1-m7, f1-f5 and croak (list them with `espeak-ng --voices=variant`)
  change the character of the voice before any effect is applied. Tune by ear.

HDMI / DISPLAYPORT AUDIO
  Needs a monitor or TV that has speakers (or an audio extractor) on the active
  display. The Jetson lists four HDMI outputs on card 0 (devices 3, 7, 8, 9) and
  only the one matching the connected port makes sound. Not sure which? Run
  `probe`: it speaks a number on every output and you note which one you hear.
  Volume is set on the monitor, not on the Jetson. If the first word is clipped
  (HDMI audio takes a moment to wake), add --lead-silence 0.5.

COMMANDS
  python3 tts_speaker.py devices          list outputs you can use
  python3 tts_speaker.py probe            speak on every output, to find the right one
  python3 tts_speaker.py say "Please put on a safety vest."
  python3 tts_speaker.py test
  python3 tts_speaker.py watch --url http://localhost:8010

"watch" polls the sentry server's /status and speaks when the verdict changes:
  - only GO and STOP are spoken; STARTING, ERROR and NO PERSON never are
  - a verdict must hold for --stable polls in a row before it is spoken, so a
    flickering result stays quiet
  - a verdict older than --max-age-ms (a stale frame) is ignored
  - after NO PERSON, the next person gets announced again
  - the server's own "message" is spoken when it has one, else a default phrase
"""

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from array import array

DEFAULT_VOICE = os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx")
MAX_CHARS = 300
DEFAULT_PHRASE = {
    "GO": "Safety vest detected. You may proceed.",
    "STOP": "No safety vest detected. Please put on a safety vest before proceeding.",
}
_voice_cache: dict = {}


_APLAY_LINE = re.compile(r"^card (\d+): (\S+) \[(.*?)\], device (\d+): (.*?)\s*(?:\[.*\])?\s*$")


def list_outputs():
    """Playback outputs from `aplay -l` as (card, device, description), leaving out
    the Jetson's internal audio fabric (APE / ADMAIF), which makes no sound."""
    try:
        text = subprocess.run(["aplay", "-l"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    outs = []
    for line in text.splitlines():
        m = _APLAY_LINE.match(line)
        if not m:
            continue
        card, short, longname, dev, name = m.groups()
        if short == "APE" or "admaif" in name.lower():
            continue
        outs.append((int(card), int(dev), f"{longname}: {name}"))
    return outs


def boost_wav(path: str, normalize: bool = True, gain: float = 1.0, target: float = 0.9) -> None:
    """Make a 16-bit WAV louder in place.

    normalize: scale so the loudest sample reaches `target` of full scale. This
               cannot clip, and it helps most for quietly-recorded voices.
    gain:      above 1, apply extra gain through a soft limiter,
               y = tanh(g*x) / tanh(g), which lifts quiet parts by about g and
               squashes peaks instead of chopping them. Below 1, plain attenuation.
    """
    with wave.open(path, "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    if params.sampwidth != 2:
        return                                   # espeak-ng and Piper both write 16-bit PCM
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()                       # WAV data is little-endian
    peak = max((abs(v) for v in samples), default=0)
    if peak == 0:
        return
    scale = (target * 32767 / peak) if normalize else 1.0
    if gain <= 1.0:
        out = array("h", (max(-32768, min(32767, int(round(v * scale * gain)))) for v in samples))
    else:
        norm = math.tanh(gain)
        out = array("h", (int(round(32767 * math.tanh(gain * (v * scale / 32767)) / norm)) for v in samples))
    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(out.tobytes())


def _read_mono16(path: str):
    """(params, samples) for a 16-bit mono WAV, else None (other formats are left untouched)."""
    with wave.open(path, "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        return None
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    return params, samples


def _write_mono16(path: str, params, samples) -> None:
    if sys.byteorder == "big":
        samples.byteswap()
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(samples.tobytes())


def deepen_wav(path: str, semitones: float) -> None:
    """Lower the pitch by resampling. Like slowing a tape, it also makes the speech a little longer."""
    if semitones == 0:
        return
    data = _read_mono16(path)
    if data is None or len(data[1]) < 2:
        return
    params, s = data
    ratio = 2 ** (semitones / 12.0)                  # above 1: longer and lower
    n_out = max(int(len(s) * ratio), 1)
    last = len(s) - 1
    out = array("h", bytes(2 * n_out))
    for i in range(n_out):
        pos = i / ratio
        i0 = int(pos)
        out[i] = s[last] if i0 >= last else int(s[i0] + (s[i0 + 1] - s[i0]) * (pos - i0))
    _write_mono16(path, params, out)


def metal_wav(path: str, freq: float = 80.0, mix: float = 0.6) -> None:
    """Ring modulation: multiply the voice by a low sine wave for a ringing, metallic timbre.
    The multiplier stays between 1 - 2*mix and 1, so the output can never clip."""
    data = _read_mono16(path)
    if data is None:
        return
    params, s = data
    w = 2 * math.pi * freq / params.framerate
    out = array("h", (int(v * ((1 - mix) + mix * math.sin(w * i))) for i, v in enumerate(s)))
    _write_mono16(path, params, out)


def lofi_wav(path: str, bits: int = 6, hold: int = 2) -> None:
    """Crunchy radio sound: keep every 2nd sample and cut the amplitude resolution to 6 bits."""
    data = _read_mono16(path)
    if data is None:
        return
    params, s = data
    step = 65536 >> bits
    out = array("h", (max(-32768, min(32767, round(s[(i // hold) * hold] / step) * step)) for i in range(len(s))))
    _write_mono16(path, params, out)


def style_wav(path: str, deeper: float, effect: str) -> None:
    deepen_wav(path, deeper)
    if effect == "metal":
        metal_wav(path)
    elif effect == "lofi":
        lofi_wav(path)


def pad_wav(path: str, seconds: float) -> None:
    """Prepend silence so HDMI audio, which needs a moment to wake, does not clip the speech."""
    with wave.open(path, "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    blank = b"\x80" if params.sampwidth == 1 else b"\x00"      # 8-bit WAV silence is 0x80
    silence = blank * (int(params.framerate * seconds) * params.nchannels * params.sampwidth)
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(silence + frames)


def clean(text: str) -> str:
    """Drop control characters, collapse whitespace and cap the length."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def have_piper(voice_path: str) -> bool:
    if not os.path.isfile(voice_path):
        return False
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return True


def choose_engine(engine: str, voice_path: str) -> str:
    if engine == "piper":
        if not os.path.isfile(voice_path):
            raise SystemExit(f"Piper voice not found: {voice_path}\n"
                             "Download one: python3 -m piper.download_voices en_US-lessac-medium")
        return "piper"
    if engine == "espeak":
        if not shutil.which("espeak-ng"):
            raise SystemExit("espeak-ng not found. Install it: sudo apt install espeak-ng")
        return "espeak"
    if have_piper(voice_path):
        return "piper"
    if shutil.which("espeak-ng"):
        return "espeak"
    raise SystemExit("No speech engine available. Install espeak-ng (sudo apt install espeak-ng) "
                     "or Piper (pip install piper-tts, plus a voice file).")


def synth_piper(text: str, voice_path: str, wav_path: str) -> None:
    from piper import PiperVoice
    voice = _voice_cache.get(voice_path)
    if voice is None:
        voice = _voice_cache[voice_path] = PiperVoice.load(voice_path)   # loaded once, reused in watch mode
    with wave.open(wav_path, "wb") as wf:
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wf)       # piper-tts 1.3 and later
        else:
            voice.synthesize(text, wf)           # older piper-tts


def synth_espeak(text: str, wav_path: str, rate: int, voice: str, pitch: int = 50) -> None:
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(rate), "-p", str(pitch), "-w", wav_path, "--", text], check=True)


PLAYER = {"name": "auto"}          # set from --player in main()
PLAY_AS = {}                       # set by flexo_outputs.py when it runs as root: {"user": uid, "group": gid, "extra_groups": [...], "env": {...}}
_INSTALL_HINT = {"aplay": "sudo apt install alsa-utils", "paplay": "sudo apt install pulseaudio-utils", "pw-play": "sudo apt install pipewire-bin"}


def pick_player(device, player: str) -> str:
    """Which program makes the sound.
    With --device the sound goes straight to that ALSA device, which only aplay can do (and which fails with
    "Device or resource busy" while the desktop's sound server holds it). Without one it goes through the
    desktop's sound server (paplay, then pw-play), which is what a logged-in desktop needs; plain aplay is the
    last resort."""
    if device:
        if player not in ("auto", "aplay"):
            raise SystemExit(f"--device names a raw ALSA device, which only aplay can use. "
                             f"Leave out --device to use {player}, or leave out --player.")
        if not shutil.which("aplay"):
            raise SystemExit("--device needs aplay. Install it: sudo apt install alsa-utils")
        return "aplay"
    for name in (["paplay", "pw-play", "aplay"] if player == "auto" else [player]):
        if shutil.which(name):
            return name
    if player != "auto":
        raise SystemExit(f"{player} was not found. Install it: {_INSTALL_HINT[player]}")
    raise SystemExit("No audio player found. Install one: sudo apt install alsa-utils pulseaudio-utils")


def play(wav_path: str, device: str | None) -> None:
    name = pick_player(device, PLAYER["name"])
    cmd = ["aplay", "-q"] + (["-D", device] if device else []) + [wav_path] if name == "aplay" else [name, wav_path]
    run_kw = {}
    if PLAY_AS:                                            # the file was made by root; the player runs as the user
        os.chmod(wav_path, 0o644)
        run_kw = dict(PLAY_AS)
    try:
        subprocess.run(cmd, check=True, **run_kw)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"{name} failed (exit {e.returncode}). Try another route with --player aplay, paplay or pw-play, "
                         "or check the output with: python3 tts_speaker.py devices   (and that nothing else is using it)")


def find_clip(clips_dir, name: str):
    """DIR/NAME.wav if it exists, else None."""
    if not clips_dir:
        return None
    path = pathlib.Path(clips_dir).expanduser() / f"{name}.wav"
    return path if path.is_file() else None


def play_clip(path, args) -> None:
    """Play a recording as it is. Only loudness and lead silence are applied, never a voice effect."""
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        shutil.copyfile(path, wav)
        try:
            if args.normalize or args.gain != 1.0:
                boost_wav(wav, args.normalize, args.gain)
            if args.lead_silence > 0:
                pad_wav(wav, args.lead_silence)
        except (wave.Error, EOFError):
            print(f"Note: {pathlib.Path(path).name} is not plain PCM WAV, so loudness and padding were skipped. "
                  "Convert it with: ffmpeg -i in.wav -ac 1 -c:a pcm_s16le out.wav")
            shutil.copyfile(path, wav)
        play(wav, args.device)
    finally:
        if os.path.exists(wav):
            os.unlink(wav)


def make_announcer(args):
    """What `watch` calls on a GO / STOP change: the recording if there is one, else the synthetic voice."""
    def announce(status, text):
        clip = find_clip(args.clips, status.lower())
        if clip:
            play_clip(clip, args)
        else:
            speak(text, args)
    return announce


def speak(text: str, args) -> None:
    text = clean(text)
    if not text:
        return
    engine = choose_engine(args.engine, args.voice)
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if engine == "piper":
            synth_piper(text, args.voice, wav)
        else:
            synth_espeak(text, wav, args.rate, args.espeak_voice, args.pitch)
        if args.deeper != 0 or args.effect != "none":
            style_wav(wav, args.deeper, args.effect)
        if args.normalize or args.gain != 1.0:
            boost_wav(wav, args.normalize, args.gain)
        if args.lead_silence > 0:
            pad_wav(wav, args.lead_silence)
        play(wav, args.device)
    finally:
        if os.path.exists(wav):
            os.unlink(wav)


def watch_loop(fetch, say, interval=0.5, stable=3, cooldown=6.0, max_age_ms=5000,
               sleep=time.sleep, now=time.monotonic, max_polls=None, log=print, announce=None,
               fresh_only=False, stop_stable=None, stop=None):
    """Speak GO / STOP when the sentry's verdict changes. fetch/say/sleep/now are
    parameters so the logic can be tested without a server, audio or real time.

    fresh_only: the VLM takes 1 to 2 s per frame but we poll every half second, so the same result is seen
                several times. With fresh_only only a NEW result (its age no larger than at the previous poll)
                counts toward `stable`, so one wrong result cannot be announced on its own. A server that
                reports no age counts every poll, as before.
    stop_stable: fresh results needed before STOP is spoken (default: the same as `stable`). GO should need
                more than STOP: a wrong GO is the dangerous mistake.
    stop: an optional threading.Event; the loop ends when it is set (pass sleep=stop.wait to wake at once)."""
    last_spoken = None
    last_time = float("-inf")
    candidate, count = None, 0
    prev_age = None
    warned = False
    polls = 0
    while (max_polls is None or polls < max_polls) and not (stop is not None and stop.is_set()):
        polls += 1
        try:
            data = fetch()
            warned = False
        except Exception as e:
            if not warned:
                log(f"cannot reach the sentry ({e}); retrying")
                warned = True
            candidate, count, prev_age = None, 0, None
            sleep(2)
            continue

        status = data.get("status")
        if status in ("GO", "STOP") and data.get("age_ms", 0) > max_age_ms:
            status = "STALE"                       # an old frame's verdict is not announced
        age = data.get("age_ms")
        fresh = (not fresh_only) or age is None or prev_age is None or age <= prev_age
        prev_age = age
        if status == candidate:
            if fresh:
                count += 1
        else:
            candidate, count = status, 1
        need = stop_stable if (status == "STOP" and stop_stable) else stable

        if count >= need:
            if status == "NO PERSON":
                last_spoken = None                 # the next person is announced again
            elif status in ("GO", "STOP") and status != last_spoken and now() - last_time >= cooldown:
                text = data.get("message") or DEFAULT_PHRASE[status]
                if announce:
                    announce(status, text)      # may play a recording for this verdict
                else:
                    say(text)
                last_spoken, last_time = status, now()
                candidate, count = None, 0
        sleep(interval)


def make_fetch(url: str, backend: str | None):
    full = url.rstrip("/") + "/status"
    if backend:
        full += "?" + urllib.parse.urlencode({"backend": backend})

    def fetch():
        with urllib.request.urlopen(full, timeout=3) as r:
            return json.load(r)
    return fetch


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Offline text-to-speech to a speaker.")
    ap.add_argument("command", choices=["devices", "probe", "say", "test", "clip", "watch"])
    ap.add_argument("text", nargs="*", help="text for 'say'")
    ap.add_argument("--engine", choices=["auto", "piper", "espeak"], default="auto")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Piper .onnx voice file")
    ap.add_argument("--device", help="ALSA output device, e.g. plughw:2,0 (default: system default)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="extra loudness through a soft limiter, e.g. 2 or 3 (default 1 = none)")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="do not scale each sentence up to near full level")
    ap.add_argument("--lead-silence", type=float, default=0.0,
                    help="seconds of silence before the speech (try 0.5 if the first word is clipped on HDMI)")
    ap.add_argument("--player", choices=["auto", "aplay", "paplay", "pw-play"], default="auto",
                    help="what plays the sound: auto uses the desktop's sound server unless --device is given")
    ap.add_argument("--clips", help="folder with go.wav and stop.wav recordings to play instead of the synthetic voice")
    ap.add_argument("--rate", type=int, default=150, help="espeak-ng words per minute")
    ap.add_argument("--pitch", type=int, default=50, help="espeak-ng pitch, 0-99 (lower is deeper)")
    ap.add_argument("--deeper", type=float, default=0.0, help="lower the voice by this many semitones (-6 to 12)")
    ap.add_argument("--effect", choices=["none", "metal", "lofi"], default="none", help="optional voice effect")
    ap.add_argument("--espeak-voice", default="en-us")
    ap.add_argument("--url", default="http://localhost:8010", help="sentry server for 'watch'")
    ap.add_argument("--backend", help="backend query for 'watch' (omit if the server is locked)")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--stable", type=int, default=3, help="fresh VLM results in a row before GO (or NO PERSON) is acted on")
    ap.add_argument("--stop-stable", type=int, default=2, help="fresh VLM results in a row before STOP is spoken")
    ap.add_argument("--count-polls", action="store_true", help="count every poll, not just new VLM results (the old behaviour)")
    ap.add_argument("--cooldown", type=float, default=6.0, help="minimum seconds between announcements")
    ap.add_argument("--max-age-ms", type=int, default=5000)
    return ap


def parse_args(argv=None):
    args = build_parser().parse_intermixed_args(argv)   # lets flags come before or after the text
    PLAYER["name"] = args.player
    if not (0 <= args.pitch <= 99):
        raise SystemExit("--pitch must be between 0 and 99.")
    if not (-6 <= args.deeper <= 12):
        raise SystemExit("--deeper must be between -6 and 12 semitones.")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.command == "devices":
        if not shutil.which("aplay"):
            raise SystemExit("aplay not found. Install it: sudo apt install alsa-utils")
        outs = list_outputs()
        if not outs:
            raise SystemExit("No playback outputs found in `aplay -l`. Plug in a USB speaker or "
                             "connect a monitor with speakers, then try again.")
        print("Outputs you can use (pass as --device):")
        for card, dev, name in outs:
            print(f"  plughw:{card},{dev}    {name}")
        print("\nOn HDMI/DisplayPort only the output matching the connected monitor makes sound. "
              "Run `probe` to hear which one.")
    elif args.command == "probe":
        outs = list_outputs()
        if not outs:
            raise SystemExit("No playback outputs found in `aplay -l`.")
        print("Speaking a number on each output. Note which one you hear.\n")
        for n, (card, dev, name) in enumerate(outs, 1):
            print(f"{n}. plughw:{card},{dev}    {name}")
            probe_args = argparse.Namespace(**vars(args))
            probe_args.device = f"plughw:{card},{dev}"
            try:
                speak(f"Output {n}. Card {card}, device {dev}.", probe_args)
            except SystemExit as e:
                print(f"   no sound path here: {e}")
            time.sleep(0.5)
        print("\nUse the line you heard, for example:  --device plughw:0,3")
    elif args.command == "test":
        speak("This is a speaker test. Safety vest detected. You may proceed.", args)
    elif args.command == "clip":
        name = " ".join(args.text).strip().lower()
        if not args.clips or not name:
            raise SystemExit("Usage: clip NAME --clips DIR   (for example: clip stop --clips ~/voice_clips)")
        path = find_clip(args.clips, name)
        if path is None:
            have = sorted(p.stem for p in pathlib.Path(args.clips).expanduser().glob("*.wav")) if pathlib.Path(args.clips).expanduser().is_dir() else []
            raise SystemExit(f"No clip named '{name}' in {args.clips}. Clips found: {', '.join(have) or 'none'}.")
        play_clip(path, args)
    elif args.command == "say":
        if not args.text:
            raise SystemExit('Give the text to say, e.g.: say "Please put on a safety vest."')
        speak(" ".join(args.text), args)
    else:
        print(f"Watching {args.url}. Ctrl+C to stop.")
        try:
            watch_loop(make_fetch(args.url, args.backend), lambda t: speak(t, args),
                       args.interval, args.stable, args.cooldown, args.max_age_ms,
                       announce=make_announcer(args), fresh_only=not args.count_polls, stop_stable=args.stop_stable)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
