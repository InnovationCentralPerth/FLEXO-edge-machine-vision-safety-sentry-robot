#!/usr/bin/env bash
# install_flexo_service.sh - ONE service for the whole sentry: Ollama, the sentry web server, the servo sign and the voice.
#
#   bash install_flexo_service.sh                     install, enable at boot and start flexo.service
#   bash install_flexo_service.sh --replace-old       also switch off the older icp-safety-sentry / -voice / -sign services
#   bash install_flexo_service.sh --enable-linger     also let the user's sound session start at boot with nobody logged in
#   bash install_flexo_service.sh --kiosk             also open the sentry page full screen in Firefox on the Jetson's screen at login
#   bash install_flexo_service.sh --no-kiosk          remove that again (the service itself is untouched)
#   bash install_flexo_service.sh --print             show the service file without installing
#   bash install_flexo_service.sh --status            is it running? last log lines, what the sentry says
#   bash install_flexo_service.sh --remove            stop it and delete the service
#
# Options (all optional; the defaults are the ones that work on this board):
#   --user NAME            the account that owns the sentry project and the desktop session (default: you)
#   --sentry-dir DIR       the sentry project (default ~USER/projects/icp_safety_vlm)
#   --backend NAME         vision backend to lock the sentry to (default vlm_ollama_ministral)
#   --model NAME           the Ollama model that backend needs, checked at start (default ministral-3:3b)
#   --no-bind-ollama       do not make Ollama part of this service (see below)
#   --kiosk-url URL        what the kiosk opens (default http://127.0.0.1:PORT/, the sentry's own page; http://host[:port][/path] only)
#   --rotate MODE          camera rotation: none, cw90, ccw90 or 180 (default ccw90)
#   --gain N               voice loudness (default 2.5)
#   --go-angle N --stop-angle N     servo positions (default 180 and 0)
#   --voice-opts "..."     more options for the voice (tts_speaker.py watch options)
#   --sign-opts "..."      more options for the sign (sign_servo.py options)
#   --no-voice  --no-sign  leave one of them out
#
# KIOSK: a system service runs as root and cannot draw on your screen, so Firefox is started by your desktop session instead: --kiosk
# puts a small launcher (flexo_kiosk.sh, in this folder) in ~/.config/autostart. It waits for the sentry that flexo.service starts, opens
# Firefox full screen, and reopens it if it is closed. It needs the desktop to log in by itself at boot (auto-login).
# Put this file in the folder that holds flexo_service.py, flexo_outputs.py, sign_servo.py, servo_pwm.py and tts_speaker.py.
# The service Requires= ollama.service and starts after it. Unless --no-bind-ollama, a drop-in also makes Ollama PART OF this service:
# stopping or restarting flexo stops or restarts Ollama too, so one command controls the lot and stopping frees the model's memory.
# Ollama stays its own unit underneath (it carries the memory setting and the model folder). The unit runs as root (the servo needs
# it); the sentry is run as --user and the voice is played as --user. It tells systemd it is READY only when the sentry answers and
# the sign and voice run, and feeds systemd's watchdog only while the sign loop is turning, so a frozen sign gets the service restarted.

set -u
UNIT_NAME="flexo.service"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"
PYTHON="${PYTHON:-/usr/bin/python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SAFE='^[A-Za-z0-9_./:@,=+~-]+$'
SAFE_SPACES='^[A-Za-z0-9_./:@,=+~ -]+$'
OLD_SERVICES=(icp-safety-sentry icp-safety-voice icp-safety-sign)

user=""; sentry_dir=""; backend="vlm_ollama_ministral"; model="ministral-3:3b"; bind=1; linger=0; kiosk=0; kiosk_url=""; rotate="ccw90"; gain="2.5"; go="180"; stop="0"; port="8010"
extra_voice=""; extra_sign=""; use_voice=1; use_sign=1; replace_old=0; mode="install"

fail() { echo "$*" >&2; exit 2; }
check() { [[ "$2" =~ $SAFE ]] || fail "$1 contains a character that is not allowed in a service file: '$2'. Use letters, digits and _ . / : @ , = + ~ -"; }
checks() { [[ "$2" =~ $SAFE_SPACES ]] || fail "$1 contains a character that is not allowed in a service file. Use letters, digits, spaces and _ . / : @ , = + ~ -"; }
number() { [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "$1 must be a number, for example 180"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --user)        [ $# -ge 2 ] || fail "--user needs a value"; user="$2"; shift ;;
    --sentry-dir)  [ $# -ge 2 ] || fail "--sentry-dir needs a value"; sentry_dir="$2"; shift ;;
    --backend)     [ $# -ge 2 ] || fail "--backend needs a value"; backend="$2"; shift ;;
    --model)       [ $# -ge 2 ] || fail "--model needs a value"; model="$2"; shift ;;
    --rotate)      [ $# -ge 2 ] || fail "--rotate needs a value"; rotate="$2"; shift ;;
    --gain)        [ $# -ge 2 ] || fail "--gain needs a value"; gain="$2"; shift ;;
    --go-angle)    [ $# -ge 2 ] || fail "--go-angle needs a value"; go="$2"; shift ;;
    --stop-angle)  [ $# -ge 2 ] || fail "--stop-angle needs a value"; stop="$2"; shift ;;
    --port)        [ $# -ge 2 ] || fail "--port needs a value"; port="$2"; shift ;;
    --voice-opts)  [ $# -ge 2 ] || fail "--voice-opts needs a value"; extra_voice="$2"; shift ;;
    --sign-opts)   [ $# -ge 2 ] || fail "--sign-opts needs a value"; extra_sign="$2"; shift ;;
    --no-voice)    use_voice=0 ;;
    --no-sign)     use_sign=0 ;;
    --replace-old) replace_old=1 ;;
    --no-bind-ollama) bind=0 ;;
    --enable-linger)  linger=1 ;;
    --kiosk)       kiosk=1 ;;
    --no-kiosk)    kiosk=2 ;;
    --kiosk-url)   [ $# -ge 2 ] || fail "--kiosk-url needs a value"; kiosk_url="$2"; kiosk=1; shift ;;
    --print)       mode="print" ;;
    --remove)      mode="remove" ;;
    --status)      mode="status" ;;
    -h|--help)     sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             fail "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

if [ "$mode" = "remove" ]; then
  $SUDO systemctl disable --now "$UNIT_NAME" >/dev/null 2>&1
  $SUDO rm -f "$UNIT_DIR/$UNIT_NAME" "$UNIT_DIR/ollama.service.d/flexo-bind.conf"
  $SUDO systemctl daemon-reload
  ru="$user"; [ -n "$ru" ] || { [ "$(id -u)" -ne 0 ] && ru="$(id -un)"; } || ru="${SUDO_USER:-}"
  if [ -n "$ru" ] && [ "$ru" != "root" ]; then
    rhome="$(getent passwd "$ru" | cut -d: -f6)"; rentry="${FLEXO_AUTOSTART_DIR:-$rhome/.config/autostart}/flexo-kiosk.desktop"
    [ ! -e "$rentry" ] || { rm -f "$rentry" 2>/dev/null || $SUDO rm -f "$rentry"; echo "Removed the kiosk autostart entry."; }
  fi
  echo "Removed $UNIT_NAME. Ollama is its own service again. The sign was parked at STOP and is held there; release it with:  sudo $PYTHON $HERE/servo_pwm.py off"
  exit 0
fi
if [ "$mode" = "status" ]; then
  systemctl status "$UNIT_NAME" --no-pager 2>&1 | head -14
  echo "--- ollama.service: $(systemctl is-active ollama.service 2>/dev/null || echo unknown)"
  echo "--- sentry says:"
  curl -fs --max-time 3 "http://127.0.0.1:$port/status" 2>/dev/null | cut -c1-200 || echo "(not answering on port $port)"
  echo; echo "--- last log lines"
  journalctl -u "$UNIT_NAME" -n 12 --no-pager 2>&1
  exit 0
fi

# ------------------------------------------------------------------ check everything before writing anything
[ -n "$user" ] || { [ "$(id -u)" -ne 0 ] && user="$(id -un)"; } || user="${SUDO_USER:-}"
[ -n "$user" ] && [ "$user" != "root" ] || fail "Give the account with --user (the one that owns the sentry project and logs in to the desktop), not root."
check "--user" "$user"; check "The folder path" "$HERE"; check "--backend" "$backend"; check "--rotate" "$rotate"; check "--model" "$model"
id -u "$user" >/dev/null 2>&1 || fail "--user '$user': no such user."
case "$rotate" in none|cw90|ccw90|180) ;; *) fail "--rotate must be none, cw90, ccw90 or 180" ;; esac
number "--gain" "$gain"; number "--go-angle" "$go"; number "--stop-angle" "$stop"; number "--port" "$port"
for a in "$go" "$stop"; do awk -v a="$a" 'BEGIN { exit !(a >= 0 && a <= 180) }' || fail "Angles must be between 0 and 180 (got $a)."; done
awk -v g="$go" -v s="$stop" 'BEGIN { exit !(g != s) }' || fail "--go-angle and --stop-angle must differ."
[ -z "$extra_voice" ] || checks "--voice-opts" "$extra_voice"
[ -z "$extra_sign" ] || checks "--sign-opts" "$extra_sign"
case " $extra_sign " in *" --min-us "*|*" --max-us "*|*" --go-us "*|*" --stop-us "*|*" --root "*|*" --dry-run "*|*" --no-off-on-exit "*)
  fail "--sign-opts cannot contain --min-us, --max-us, --go-us, --stop-us, --root, --dry-run or --no-off-on-exit: the service parks the sign with servo_pwm.py at the default range, so they would disagree." ;;
esac
[ "$use_voice" -eq 1 ] || [ "$use_sign" -eq 1 ] || true            # the sentry alone is allowed
uid="$(id -u "$user")"
home="$(getent passwd "$user" | cut -d: -f6)"
[ -n "$kiosk_url" ] || kiosk_url="http://127.0.0.1:$port/"
[[ "$kiosk_url" =~ ^http://[A-Za-z0-9._-]+(:[0-9]+)?(/[A-Za-z0-9._~/-]*)?$ ]] || fail "--kiosk-url must be plain http://host[:port][/path] (letters, digits and . _ - / only; no query string): '$kiosk_url'"
autostart_dir="${FLEXO_AUTOSTART_DIR:-$home/.config/autostart}"; kiosk_entry="$autostart_dir/flexo-kiosk.desktop"
[ "$kiosk" -ne 1 ] || [ -f "$HERE/flexo_kiosk.sh" ] || fail "flexo_kiosk.sh is not in $HERE. Put it next to this script."
[ -n "$sentry_dir" ] || sentry_dir="$home/projects/icp_safety_vlm"
[ -z "$sentry_dir" ] || check "--sentry-dir" "$sentry_dir"
[ -f "$sentry_dir/pyproject.toml" ] || fail "The sentry project is not at $sentry_dir (use --sentry-dir)."
[ -d "$sentry_dir/.venv" ] || fail "$sentry_dir/.venv is missing: run  cd $sentry_dir && uv sync  as $user."
[ -x "${FLEXO_UV:-$home/.local/bin/uv}" ] || fail "uv was not found at ${FLEXO_UV:-$home/.local/bin/uv}."
for f in flexo_service.py flexo_outputs.py sign_servo.py servo_pwm.py tts_speaker.py; do
  [ -f "$HERE/$f" ] || fail "$f is not in $HERE. Put this script next to it."
done
[ -x "$PYTHON" ] || fail "$PYTHON not found. Set PYTHON=/path/to/python3."
systemctl cat ollama.service >/dev/null 2>&1 || fail "ollama.service does not exist on this machine, so there is nothing for flexo.service to depend on."

# the older boot services do the same jobs and would fight this one
old_found=()
for s in "${OLD_SERVICES[@]}"; do
  if [ "$(systemctl is-enabled "$s" 2>/dev/null)" = "enabled" ] || [ "$(systemctl is-active "$s" 2>/dev/null)" = "active" ]; then old_found+=("$s"); fi
done
if [ ${#old_found[@]} -gt 0 ] && [ "$replace_old" -eq 0 ] && [ "$mode" != "print" ]; then
  echo "These older services are enabled or running and do the same jobs: ${old_found[*]}" >&2
  fail "Run again with --replace-old to switch them off (they are not deleted; you can enable them again), or remove them yourself."
fi

# ------------------------------------------------------------------ the service file
args=(--user "$user" --sentry-dir "$sentry_dir" --backend "$backend" --rotate "$rotate" --gain "$gain" --go-angle "$go" --stop-angle "$stop" --port "$port" --model "$model")
[ -z "$extra_voice" ] || args+=(--voice-opts "\"$extra_voice\"")
[ -z "$extra_sign" ] || args+=(--sign-opts "\"$extra_sign\"")
[ "$use_voice" -eq 1 ] || args+=(--no-voice)
[ "$use_sign" -eq 1 ] || args+=(--no-sign)
stop_post=""
[ "$use_sign" -eq 0 ] || stop_post="ExecStopPost=-$PYTHON $HERE/servo_pwm.py --period-ms 5 --wide angle $stop
"

unit="[Unit]
Description=ICP FLEXO safety sentry: Ollama, sentry web server, servo sign and voice, as one service
Requires=ollama.service
After=ollama.service network-online.target sound.target user@$uid.service
Wants=network-online.target user@$uid.service
Conflicts=${OLD_SERVICES[0]}.service ${OLD_SERVICES[1]}.service ${OLD_SERVICES[2]}.service
StartLimitIntervalSec=300
StartLimitBurst=8

[Service]
Type=notify
NotifyAccess=main
User=root
WorkingDirectory=$HERE
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
SyslogIdentifier=flexo
ExecStart=$PYTHON $HERE/flexo_service.py ${args[*]}
${stop_post}KillMode=mixed
TimeoutStartSec=420
TimeoutStopSec=45
WatchdogSec=15
Restart=always
RestartSec=5
RestartPreventExitStatus=2

[Install]
WantedBy=multi-user.target
"
if [ "$mode" = "print" ]; then printf '%s' "$unit"; exit 0; fi

# ------------------------------------------------------------------ warnings that do not stop the install
if ! systemctl show ollama -p Environment 2>/dev/null | grep -q "LLAMA_ARG_CACHE_RAM=0"; then
  echo "Warning: Ollama does not have LLAMA_ARG_CACHE_RAM=0. Without it its memory grows until the kernel kills it (the guide, section 7)."
fi
if [ ! -d "/run/user/$uid" ] && ! loginctl show-user "$user" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
  echo "Note: $user has no desktop session running, so the voice has nowhere to play until $user logs in. Log in automatically, or:  sudo loginctl enable-linger $user"
fi
[ "$use_sign" -eq 0 ] || [ -d /sys/class/pwm ] || echo "Warning: no /sys/class/pwm here. Route pin 33 to pwm5 with jetson-io and reboot first; the service will refuse to start without it."

# ------------------------------------------------------------------ install
chmod +x "$HERE/flexo_service.py" 2>/dev/null
if [ "$replace_old" -eq 1 ]; then
  for s in "${old_found[@]+"${old_found[@]}"}"; do $SUDO systemctl disable --now "$s" >/dev/null 2>&1 && echo "Switched off the older service $s."; done
fi
dropin="[Unit]
# makes Ollama part of flexo.service: stopping or restarting flexo also stops or restarts Ollama (written by install_flexo_service.sh)
PartOf=$UNIT_NAME
"
[ "$linger" -eq 0 ] || $SUDO loginctl enable-linger "$user" 2>&1 | head -2
printf '%s' "$unit" | $SUDO tee "$UNIT_DIR/$UNIT_NAME" >/dev/null || fail "Could not write $UNIT_DIR/$UNIT_NAME"
if [ "$bind" -eq 1 ]; then
  $SUDO mkdir -p "$UNIT_DIR/ollama.service.d"
  printf '%s' "$dropin" | $SUDO tee "$UNIT_DIR/ollama.service.d/flexo-bind.conf" >/dev/null || fail "Could not write the Ollama drop-in"
else
  $SUDO rm -f "$UNIT_DIR/ollama.service.d/flexo-bind.conf"
fi
as_user() { if [ "$(id -u)" -eq "$uid" ]; then "$@"; elif [ "$(id -u)" -eq 0 ]; then runuser -u "$user" -- "$@"; else sudo -u "$user" "$@"; fi; }
if [ "$kiosk" -eq 1 ]; then
  chmod +x "$HERE/flexo_kiosk.sh" 2>/dev/null
  as_user mkdir -p "$autostart_dir" || fail "could not create $autostart_dir"
  printf '%s\n' "[Desktop Entry]
Type=Application
Name=FLEXO sentry screen
Comment=Opens the sentry page full screen once the sentry answers (written by install_flexo_service.sh)
Exec=env FLEXO_KIOSK_URL=$kiosk_url $HERE/flexo_kiosk.sh
Terminal=false
NoDisplay=true
X-GNOME-Autostart-enabled=true" | as_user tee "$kiosk_entry" >/dev/null || fail "could not write $kiosk_entry"
  echo "Kiosk: Firefox will open $kiosk_url full screen when $user's desktop starts."
  echo "       To start it right now, in a terminal ON the Jetson's desktop:  FLEXO_KIOSK_URL=$kiosk_url $HERE/flexo_kiosk.sh &"
elif [ "$kiosk" -eq 2 ]; then
  as_user rm -f "$kiosk_entry" && echo "Removed the kiosk autostart entry."
fi
$SUDO systemctl daemon-reload
echo "Starting $UNIT_NAME: it waits for Ollama and the model, so the first start can take a few minutes..."
$SUDO systemctl enable --now "$UNIT_NAME" 2>&1 | tail -1
echo "Service $UNIT_NAME is: $(systemctl is-active "$UNIT_NAME" 2>/dev/null || echo unknown)"
echo "Order: Ollama, then the sentry, then the sign and voice. systemd calls it active only when all of them are running."
[ "$bind" -eq 0 ] || echo "Stopping flexo also stops Ollama and frees its memory; starting flexo starts it again.  (Not wanted? reinstall with --no-bind-ollama)"
echo "Watch it:  journalctl -u flexo -f        Check later:  bash $0 --status        Dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$port/"
echo "When it stops, the sign is parked at STOP and held there."
