FLEXO: ONE service for Ollama + the sentry + the GO/STOP servo sign + the voice
================================================================================

WHERE TO PUT IT: any one folder. The service finds its scripts through the folder the installer is in, so you can put these
files straight into ~/FLEXO-edge-machine-vision-safety-sentry-robot/ (the folder path may not contain spaces). Run the installer
from there. If you move the folder later, run the installer again so the service file points at the new place.
  * tts_speaker.py: the one here is newer than the one you have; overwrite yours.
  * sign_servo.py and servo_pwm.py here replace the ones in ~/servo_test; use these, delete the old copies.
  * The service runs as root but writes no __pycache__ (PYTHONDONTWRITEBYTECODE is set). If you run these by hand with sudo, add
    __pycache__/ to .gitignore.
  * The sentry project (~/projects/icp_safety_vlm) can stay where it is. Do not just move it: its .venv has absolute paths and
    breaks; if you do move it, run  uv sync  in the new place and pass  --sentry-dir NEWPATH  to the installer.

BEFORE YOU INSTALL (once)
  * Ollama is installed with its memory setting (LLAMA_ARG_CACHE_RAM=0) and the model is pulled:  ollama list   shows ministral-3:3b
  * The sentry project builds:  cd ~/projects/icp_safety_vlm && uv sync
  * Pin 33 is routed to pwm5 (jetson-io) and the servo works by hand:  sudo python3 servo_pwm.py --period-ms 5 --wide angle 0
  * Stop anything you started by hand (the sentry on port 8010, tts_speaker.py, sign_servo.py): they would fight the service.

INSTALL (as your normal user, not with sudo; it asks for your password itself)
  cd ~/FLEXO-edge-machine-vision-safety-sentry-robot      # or wherever you put these files
  bash install_flexo_service.sh --print          # look at the service file first (changes nothing)
  bash install_flexo_service.sh --replace-old    # install, enable at boot, start. --replace-old only matters if the older
                                                 # icp-safety-sentry / -voice / -sign services exist: it switches them off.
  If your sign shows GO at the other end of the servo:   add  --go-angle 0 --stop-angle 180

FIREFOX ON THE JETSON'S SCREEN (the kiosk)
  bash install_flexo_service.sh --kiosk                  also open the sentry page full screen at login
  bash install_flexo_service.sh --kiosk-url http://192.168.1.50:8010/      another address (default: http://127.0.0.1:8010/)
  bash install_flexo_service.sh --no-kiosk               take it out again
  Why it is not part of flexo.service: a system service runs as root and cannot draw on your screen. Your desktop session can, so the
  installer puts flexo_kiosk.sh in ~/.config/autostart. It waits for the sentry that flexo.service starts, opens Firefox with
  --kiosk, and reopens it if it is closed. It does not open a second Firefox if one is already running.
  It needs the desktop to log in by itself at boot (auto-login). Start it now without rebooting, in a terminal ON the Jetson:
      FLEXO_KIOSK_URL=http://127.0.0.1:8010/ ./flexo_kiosk.sh &
  Stop it:  pkill -f flexo_kiosk.sh  and close Firefox.   Its log:  ~/.cache/flexo-kiosk.log
  The address is for the Jetson's OWN screen, so 127.0.0.1 is right and keeps working if the network address changes. Use the IP only
  if you want this screen to show a different machine's sentry.
  patch_feed_fit.py (also in this folder) makes the camera picture bigger on this page: python3 patch_feed_fit.py

DAY TO DAY
  journalctl -u flexo -f                  watch it ([main] [sentry] [sign] [voice] tags)
  bash install_flexo_service.sh --status
  sudo systemctl stop flexo               stops EVERYTHING: the sign is parked at STOP, the sentry stops, Ollama stops (frees its memory)
  sudo systemctl start flexo              starts it all again, in order. 'start' returns only when everything is really running.
  sudo systemctl disable flexo            do not start at boot        sudo systemctl enable flexo   start at boot again
  bash install_flexo_service.sh --remove  take it out (the servo is left holding STOP:  sudo python3 servo_pwm.py off  to release it)

WHAT IT DOES
  Order:   Ollama answers and the model is present -> camera is there -> sentry starts (as your user) and answers -> sign + voice start -> READY.
  Sign:    GO only after 3 NEW GO results in a row; anything else, or a sentry that does not answer, shows STOP. Every stop or crash parks it at STOP.
  Voice:   plays as your user. A voice problem is logged and never affects the sign.
  Sentry:  if it dies it is restarted (the sign shows STOP meanwhile); if it keeps dying (5 times in 2 minutes) the service gives up and systemd restarts it all.
  Watchdog: systemd is pinged only while the sign loop is turning. If the sign freezes, systemd restarts the service (within about 20 s) and the stop
           step parks the sign at STOP.
  Exit codes: 1 = might fix itself (restarted automatically). 2 = a set-up problem (no model, no PWM chip, no such user, an old service still running):
           NOT restarted; read the message with  journalctl -u flexo -n 30.

OPTIONS (installer)  --user --sentry-dir --backend --model --rotate --port --gain --go-angle --stop-angle --voice-opts "..." --sign-opts "..."
                     --no-voice --no-sign --no-bind-ollama (keep Ollama independent) --enable-linger (user's sound session at boot) --replace-old

VOICE AT BOOT needs the user's sound session. The unit asks systemd to start it (user@UID.service); if the voice is silent after a reboot with
nobody logged in, reinstall with --enable-linger.

TESTED WITH FAKES (a fake Ollama, sentry, PWM and speaker; real espeak; systemd-analyze verify): start-up order and READY timing, the watchdog stopping and
resuming on a frozen sign loop, a killed / crash-looping / silent / non-starting sentry, missing model / camera / PWM, the older services, running as another
user, clean SIGTERM, the installer and the exact line it writes.
NOT TESTED: a real boot, real systemd start/stop and watchdog, the real Ollama drop-in, real audio at boot, the real servo with the service. Check the first
start with  journalctl -u flexo -f  and report anything odd.
