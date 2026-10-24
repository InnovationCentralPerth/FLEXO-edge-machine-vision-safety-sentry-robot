#!/usr/bin/env bash
# install_voice_service.sh - read the sentry's verdict aloud, started at boot (a systemd service).
#
#   bash install_voice_service.sh --device plughw:0,3                 install, enable and start it
#   bash install_voice_service.sh --device plughw:0,3 --gain 3 --backend vlm_ollama_ministral
#   bash install_voice_service.sh --device plughw:0,3 --clips ~/voice_clips   play recordings instead of the synthetic voice
#   bash install_voice_service.sh --print --device plughw:0,3         show the service file without installing
#   bash install_voice_service.sh --status                            is it running? last log lines
#   bash install_voice_service.sh --remove                            stop it and delete it
#   anything after a lone  --  is passed to tts_speaker.py watch:   ... -- --effect metal --deeper 2 --rate 170
#
# Put this file in the FLEXO folder, next to tts_speaker.py, and build the environment first
# (bash start_venv.sh --setup). The service runs tts_speaker.py with the environment's own python,
# so it needs no activation. Options: --url (default http://127.0.0.1:8010), --backend, --device,
# --gain (default 2.5), --lead-silence (default 0.5), --clips.
#
# Which route the sound takes:
#   no --device   through the logged-in user's sound server (PipeWire/PulseAudio), the same route as a normal desktop
#                 app. This is the one to use while the desktop is running. The service needs that user to be logged
#                 in (auto-login is fine); until then it keeps retrying every few seconds.
#   --device X    straight to that ALSA device. Only for a board with no desktop sound server: with the desktop
#                 running, the log says "Device or resource busy".
# See the log with:  journalctl -u icp-safety-voice -n 20 --no-pager

set -u
UNIT_NAME="icp-safety-voice.service"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SAFE='^[A-Za-z0-9_./:@,=+~-]+$'          # no spaces, quotes, $, %, ; or newlines: nothing a unit file could misread

url="http://127.0.0.1:8010"; backend=""; device=""; gain="2.5"; lead="0.5"; clips=""
mode="install"; extras=()

fail() { echo "$*" >&2; exit 2; }
check() { [[ "$2" =~ $SAFE ]] || fail "$1 contains a character that is not allowed in a service file: '$2'. Use letters, digits and _ . / : @ , = + ~ -"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --device)       [ $# -ge 2 ] || fail "--device needs a value, for example plughw:0,3"; device="$2"; shift ;;
    --backend)      [ $# -ge 2 ] || fail "--backend needs a value"; backend="$2"; shift ;;
    --url)          [ $# -ge 2 ] || fail "--url needs a value"; url="$2"; shift ;;
    --gain)         [ $# -ge 2 ] || fail "--gain needs a value"; gain="$2"; shift ;;
    --lead-silence) [ $# -ge 2 ] || fail "--lead-silence needs a value"; lead="$2"; shift ;;
    --clips)        [ $# -ge 2 ] || fail "--clips needs a folder"; clips="$2"; shift ;;
    --print)        mode="print" ;;
    --remove)       mode="remove" ;;
    --status)       mode="status" ;;
    -h|--help)      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --)             shift; extras=("$@"); break ;;
    *)              fail "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

if [ "$mode" = "remove" ]; then
  $SUDO systemctl disable --now "$UNIT_NAME" >/dev/null 2>&1
  $SUDO rm -f "$UNIT_DIR/$UNIT_NAME"
  $SUDO systemctl daemon-reload
  echo "Removed $UNIT_NAME."
  exit 0
fi
if [ "$mode" = "status" ]; then
  systemctl status "$UNIT_NAME" --no-pager 2>&1 | head -12
  echo "--- last log lines"
  journalctl -u "$UNIT_NAME" -n 8 --no-pager 2>&1
  exit 0
fi

# ---- everything that goes into the service file must be safe, and the pieces must exist
check "The FLEXO folder path" "$HERE"
check "--url" "$url"; check "--gain" "$gain"; check "--lead-silence" "$lead"
[ -n "$device" ]  && check "--device" "$device"
[ -n "$backend" ] && check "--backend" "$backend"
[ -n "$clips" ]   && check "--clips" "$clips"
for x in "${extras[@]+"${extras[@]}"}"; do check "an option after --" "$x"; done
[[ "$gain" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "--gain must be a number, for example 2.5"
[[ "$lead" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "--lead-silence must be a number of seconds, for example 0.5"

PY="$HERE/vlm-env/bin/python"
[ -x "$PY" ] || fail "No environment at $PY. Build it first:  bash $HERE/start_venv.sh --setup"
[ -f "$HERE/tts_speaker.py" ] || fail "tts_speaker.py is not in $HERE. Put it next to this script."

args=(watch --url "$url")
[ -n "$backend" ] && args+=(--backend "$backend")
[ -n "$device" ]  && args+=(--device "$device")
args+=(--gain "$gain" --lead-silence "$lead")
[ -n "$clips" ]   && args+=(--clips "$clips")
args+=("${extras[@]+"${extras[@]}"}")

audio_env=""
if [ -z "$device" ]; then
  audio_env="Environment=XDG_RUNTIME_DIR=/run/user/%U
Environment=PULSE_SERVER=unix:/run/user/%U/pulse/native
"
fi

unit="[Unit]
Description=ICP FLEXO sentry: read the verdict aloud
After=icp-safety-sentry.service sound.target
Wants=icp-safety-sentry.service

[Service]
Type=simple
User=$(id -un)
SupplementaryGroups=audio
WorkingDirectory=$HERE
Environment=PYTHONUNBUFFERED=1
${audio_env}ExecStart=$PY $HERE/tts_speaker.py ${args[*]}
Restart=always
RestartSec=3
Nice=5

[Install]
WantedBy=multi-user.target
"

if [ "$mode" = "print" ]; then printf '%s' "$unit"; exit 0; fi

# ---- warnings only: a missing engine or player is worth knowing about before the first announcement
if [ -z "$clips" ] && ! command -v espeak-ng >/dev/null 2>&1 && [ ! -f "$HOME/piper-voices/en_US-lessac-medium.onnx" ]; then
  echo "Warning: no speech engine found. Install one:  sudo apt install -y espeak-ng"
fi
command -v aplay >/dev/null 2>&1 || echo "Warning: aplay is missing:  sudo apt install -y alsa-utils"
curl -fs --max-time 2 "$url/status" >/dev/null 2>&1 || echo "Note: the sentry is not answering at $url yet. The service keeps retrying, so that is fine if it starts later."

printf '%s' "$unit" | $SUDO tee "$UNIT_DIR/$UNIT_NAME" >/dev/null || fail "Could not write $UNIT_DIR/$UNIT_NAME"
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now "$UNIT_NAME" 2>&1 | tail -1
sleep 2
state="$(systemctl is-active "$UNIT_NAME" 2>/dev/null)"
echo "Service $UNIT_NAME is: ${state:-unknown}"
echo "Command it runs:  $PY $HERE/tts_speaker.py ${args[*]}"
echo "See what it says:  journalctl -u icp-safety-voice -f      Check later:  bash $0 --status"
