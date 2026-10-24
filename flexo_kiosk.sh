#!/usr/bin/env bash
# flexo_kiosk.sh - open the sentry's page full screen in Firefox on THIS screen, and keep it open.
#
# It is started by the desktop session (install_flexo_service.sh --kiosk puts it in ~/.config/autostart), not by the flexo service:
# a system service runs as root with no access to your screen, whereas the desktop session has the display, the Wayland/X
# socket and your sound/dbus environment already set up. This script waits for the sentry that flexo.service starts, so the
# order does not matter.
#
#   ./flexo_kiosk.sh                                  open http://127.0.0.1:8010/
#   FLEXO_KIOSK_URL=http://192.168.1.50:8010/ ./flexo_kiosk.sh      another address (the Jetson's own IP also works)
#
# Run it from a terminal ON the Jetson's desktop, not over SSH (there is no screen there).
# Closing Firefox reopens it after a few seconds. To stop it for good:  pkill -f flexo_kiosk.sh  and close Firefox, or
# remove the autostart entry with  bash install_flexo_service.sh --no-kiosk.
# Log: ~/.cache/flexo-kiosk.log

set -u
URL="${FLEXO_KIOSK_URL:-http://127.0.0.1:8010/}"
LOG="${FLEXO_KIOSK_LOG:-$HOME/.cache/flexo-kiosk.log}"
FIREFOX="${FLEXO_KIOSK_FIREFOX:-firefox}"
POLL="${FLEXO_KIOSK_POLL:-2}"                    # seconds between asking whether the sentry answers
RETRY="${FLEXO_KIOSK_RETRY:-3}"                  # seconds before reopening a browser that was closed
QUICK_SECS="${FLEXO_KIOSK_QUICK_SECS:-5}"        # a Firefox that ends sooner than this did not really start
QUICK_DELAY="${FLEXO_KIOSK_QUICK_DELAY:-15}"     # ... so wait this long before trying again
MAX="${FLEXO_KIOSK_MAX_LAUNCHES:-0}"             # test hook: stop after this many launches (0 = never)
X11_DIR="${FLEXO_KIOSK_X11_DIR:-/tmp/.X11-unix}"

mkdir -p "$(dirname "$LOG")" 2>/dev/null
say() { local m; m="$(date '+%F %T') $*"; echo "$m"; echo "$m" >> "$LOG" 2>/dev/null; }

[[ "$URL" =~ ^http://[A-Za-z0-9._-]+(:[0-9]+)?(/[A-Za-z0-9._~/-]*)?$ ]] || { say "refusing the URL '$URL': only plain http://host[:port][/path] is accepted"; exit 2; }
BASE="$(printf '%s' "$URL" | sed -E 's#^(http://[^/]+).*#\1#')"
STATUS="$BASE/status"

# a screen to draw on
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  if [ -e "$X11_DIR/X0" ]; then export DISPLAY=":0"; else
    say "no screen found (no DISPLAY or WAYLAND_DISPLAY). Run this in a terminal on the Jetson's own desktop, or let the desktop autostart it."
    exit 2
  fi
fi

sentry_up() { curl -fs --max-time 2 "$STATUS" 2>/dev/null | grep -q '"status"'; }
firefox_running() { pgrep -u "$(id -u)" -x "$(basename "$FIREFOX")|firefox-bin|firefox-esr" >/dev/null 2>&1; }   # its process name depends on how it was installed

wait_for_sentry() {
  local warned=0
  until sentry_up; do
    if [ "$warned" -eq 0 ]; then say "waiting for the sentry at $STATUS"; warned=1; fi
    sleep "$POLL"
  done
}

n=0
while :; do
  wait_for_sentry
  if firefox_running; then
    say "another Firefox is already running; not opening a second one (waiting for it to close)"
    while firefox_running; do sleep 5; done
  fi
  n=$((n + 1)); start="$(date +%s)"
  say "opening Firefox full screen at $URL"
  "$FIREFOX" --kiosk "$URL" >> "$LOG" 2>&1
  rc=$?; took=$(( $(date +%s) - start ))
  say "Firefox ended (code $rc) after ${took}s"
  if [ "$MAX" -gt 0 ] && [ "$n" -ge "$MAX" ]; then break; fi
  if [ "$took" -lt "$QUICK_SECS" ]; then
    say "that was too quick to be a normal close; trying again in ${QUICK_DELAY}s"
    sleep "$QUICK_DELAY"
  else
    sleep "$RETRY"
  fi
done
