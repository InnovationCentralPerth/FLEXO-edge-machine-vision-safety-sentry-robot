#!/usr/bin/env bash
# start_venv.sh - the FLEXO project's Python virtual environment, kept INSIDE this folder.
#
# Put this file in the FLEXO folder (next to benchmark_test/). The environment is built in
# ./vlm-env, so nothing lives in your home directory.
#
#   bash   start_venv.sh --setup             build ./vlm-env if missing, install the packages
#   source start_venv.sh                     activate it in THIS terminal (builds it first if missing)
#   bash   start_venv.sh --autostart         every new terminal activates it from now on
#   bash   start_venv.sh --remove-autostart  undo that
#   bash   start_venv.sh --status            what is set up, and what is active
#   bash   start_venv.sh --recreate          delete ./vlm-env and build it again
#   source start_venv.sh --quiet             activate silently if it exists (this is what ~/.bashrc runs)
#
# Packages: the lines in requirements-vlm.txt next to this script if that file exists, otherwise
# ollama, opencv-python-headless and numpy.
#
# "Activate" only works with `source` (or `.`): a program cannot change the terminal that started
# it. Services (systemd), cron and `sudo` do not read ~/.bashrc and do not see an activated
# environment; give them the environment's own python instead:  ./vlm-env/bin/python script.py

_flexo_sourced=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then _flexo_sourced=1; fi
_flexo_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

_flexo_usage() { sed -n '2,19p' "$_flexo_here/start_venv.sh" | sed 's/^# \{0,1\}//'; }

_flexo_setup() {
  local venv="$1"
  if [ -d "$venv" ] && [ ! -x "$venv/bin/python" ]; then
    echo "$venv exists but is not a working environment. Rebuild it with:  bash $_flexo_here/start_venv.sh --recreate"
    return 1
  fi
  if [ ! -d "$venv" ]; then
    echo "Creating $venv ..."
    if ! python3 -m venv "$venv"; then
      rm -rf "$venv"
      echo "Could not create it. On Ubuntu you may need:  sudo apt install -y python3-venv"
      return 1
    fi
  fi
  local -a pkgs
  if [ -f "$_flexo_here/requirements-vlm.txt" ]; then
    pkgs=(-r "$_flexo_here/requirements-vlm.txt")
  else
    pkgs=(ollama opencv-python-headless numpy)
  fi
  echo "Installing: ${pkgs[*]}"
  if ! "$venv/bin/python" -m pip install "${pkgs[@]}"; then
    echo "Package install failed. Check the network, then run this again (it resumes)."
    return 1
  fi
  "$venv/bin/python" -c "import ollama" 2>/dev/null && echo "Check: the ollama package imports fine."
  echo "Environment ready: $venv"
  return 0
}

_flexo_activate() {
  local venv="$1" quiet="$2"
  if [ ! -f "$venv/bin/activate" ]; then
    if [ "$quiet" = 1 ]; then
      echo "FLEXO environment not built yet. Run:  bash $_flexo_here/start_venv.sh --setup"
      return 0
    fi
    _flexo_setup "$venv" || return 1
  fi
  if [ "${VIRTUAL_ENV:-}" = "$venv" ]; then
    [ "$quiet" = 1 ] || echo "Already active: $venv"
    return 0
  fi
  if [ -n "${VIRTUAL_ENV:-}" ] && type deactivate >/dev/null 2>&1; then deactivate; fi   # leave any other environment first
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  [ "$quiet" = 1 ] || echo "Active: $VIRTUAL_ENV  ($(python --version 2>&1))"
  return 0
}

_flexo_autostart_add() {
  local rc="$HOME/.bashrc" begin="# >>> flexo venv >>>" end="# <<< flexo venv <<<"
  if grep -qF "$begin" "$rc" 2>/dev/null; then echo "Already set up in $rc"; return 0; fi
  if [ -f "$rc" ]; then
    [ -f "$rc.bak-flexo" ] || cp "$rc" "$rc.bak-flexo"            # one-time backup of the original
    [ -n "$(tail -c1 "$rc")" ] && echo >> "$rc"                   # make sure the last line ends with a newline
  fi
  {
    printf '%s\n' "$begin"
    printf '%s\n' "[ -f \"$_flexo_here/start_venv.sh\" ] && source \"$_flexo_here/start_venv.sh\" --quiet"
    printf '%s\n' "$end"
  } >> "$rc"
  echo "Added to $rc (backup of the original: $rc.bak-flexo)."
  echo "New terminals will activate the environment. In this one, run:  source $rc"
}

_flexo_autostart_remove() {
  local rc="$HOME/.bashrc"
  if ! grep -qF "# >>> flexo venv >>>" "$rc" 2>/dev/null; then echo "Nothing to remove in $rc"; return 0; fi
  sed -i '/# >>> flexo venv >>>/,/# <<< flexo venv <<</d' "$rc"
  echo "Removed from $rc. New terminals will not activate it."
}

_flexo_status() {
  local venv="$1"
  echo "Folder:        $_flexo_here"
  if [ -x "$venv/bin/python" ]; then echo "Environment:   $venv ($("$venv/bin/python" --version 2>&1))"
  else echo "Environment:   not built (run: bash $_flexo_here/start_venv.sh --setup)"; fi
  echo "Active now:    ${VIRTUAL_ENV:-none in this terminal}"
  if grep -qF "# >>> flexo venv >>>" "$HOME/.bashrc" 2>/dev/null; then echo "Autostart:     on (new terminals activate it)"
  else echo "Autostart:     off (bash $_flexo_here/start_venv.sh --autostart)"; fi
  if [ -x "$venv/bin/python" ]; then
    "$venv/bin/python" - <<'PY'
for name in ("ollama", "cv2", "numpy"):
    try:
        __import__(name); print(f"Package:       {name} ok")
    except Exception as e:
        print(f"Package:       {name} MISSING ({type(e).__name__})")
PY
  fi
}

_flexo_main() {
  local venv="${FLEXO_VENV:-$_flexo_here/vlm-env}" mode="activate" quiet=0 arg
  for arg in "$@"; do
    case "$arg" in
      --quiet) quiet=1 ;;
      --setup) mode="setup" ;;
      --recreate) mode="recreate" ;;
      --autostart) mode="autostart" ;;
      --remove-autostart) mode="unautostart" ;;
      --status) mode="status" ;;
      -h|--help) _flexo_usage; return 0 ;;
      *) echo "Unknown option: $arg (try --help)"; return 2 ;;
    esac
  done

  case "$mode" in
    setup)       _flexo_setup "$venv"; return $? ;;
    recreate)    rm -rf "$venv"; _flexo_setup "$venv"; return $? ;;
    autostart)   _flexo_autostart_add; return $? ;;
    unautostart) _flexo_autostart_remove; return $? ;;
    status)      _flexo_status "$venv"; return 0 ;;
  esac

  # mode: activate
  if [ "$_flexo_sourced" != 1 ]; then
    echo "This script has to be SOURCED to activate the environment in your terminal:"
    echo "    source $_flexo_here/start_venv.sh"
    echo "(To build it, or to turn on autostart, run it with bash: see --help.)"
    return 2
  fi
  case $- in *i*) ;; *) [ "$quiet" = 1 ] && return 0 ;; esac    # quiet + non-interactive shell: do nothing
  _flexo_activate "$venv" "$quiet"
}

_flexo_main "$@"
_flexo_rc=$?
if [ "$_flexo_sourced" = 1 ]; then return "$_flexo_rc"; else exit "$_flexo_rc"; fi
