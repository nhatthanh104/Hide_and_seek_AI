#!/usr/bin/env bash
# Run multiple Arena games and summarize results.
# Default runs = 100 (change RUNS below or pass first arg to override)

# Configuration (edit here)
SEEKER="${SEEKER:-example_student}"             # Pacman student id
HIDER="${HIDER:-example_student}"               # Ghost student id
RUNS="${RUNS:-100}"                             # Default number of games
NO_VIZ=true                                     # true -> pass --no-viz
SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-../submissions}"
STEP_TIMEOUT="${STEP_TIMEOUT:-3.0}"
LOGFILE="${LOGFILE:-test_run.log}"               # File to store full run logs

# Auto-detect Python command
PYTHON_CMD=""
for python_path in \
    "/c/Users/$USER/miniconda3/envs/ml/python.exe" \
    "/c/Users/$USER/anaconda3/envs/ml/python.exe" \
    "$HOME/miniconda3/envs/ml/python.exe" \
    "$HOME/anaconda3/envs/ml/python.exe"; do
    if [ -f "$python_path" ]; then
        PYTHON_CMD="$python_path"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python not found!" >&2
        exit 1
    fi
fi

# You can override RUNS by passing a first argument to the script
if [ -n "$1" ]; then
  RUNS="$1"
fi

pacman_wins=0
ghost_wins=0
draws=0
errors=0

# Initialize logfile
echo "Run started: $(date)" > "$LOGFILE"
echo "Matchup: Pacman='$SEEKER' vs Ghost='$HIDER'" >> "$LOGFILE"
echo "Runs: $RUNS" >> "$LOGFILE"
echo "--" >> "$LOGFILE"

## NOTE: This script writes full per-game output to "$LOGFILE".
## The console will only display the final summary (total games, pacman wins, ghost wins).

for i in $(seq 1 "$RUNS"); do
  # run arena.py from src directory, capture stdout+stderr
  export PYTHONIOENCODING=utf-8
  output="$(cd src && $PYTHON_CMD arena.py --seek "$SEEKER" --hide "$HIDER" --submissions-dir "$SUBMISSIONS_DIR" --step-timeout "$STEP_TIMEOUT" $( [ "$NO_VIZ" = true ] && echo --no-viz ) 2>&1)"

  # Append full output to logfile with header
  echo "=== Game $i / $RUNS : $(date) ===" >> "$LOGFILE"
  echo "$output" >> "$LOGFILE"
  echo "" >> "$LOGFILE"

  # parse result (look for winner markers in the game's output)
  if echo "$output" | grep -q "(Pacman)"; then
    pacman_wins=$((pacman_wins+1))
  elif echo "$output" | grep -q "(Ghost)"; then
    ghost_wins=$((ghost_wins+1))
  elif echo "$output" | grep -qi "draw\|DRAW\|🤝"; then
    draws=$((draws+1))
  else
    errors=$((errors+1))
    # Save failing output for inspection (also appended to logfile)
    echo "[ERROR] Could not determine winner for game $i" >> "$LOGFILE"
  fi
done

## Print only the requested summary to console
echo "Total games: $RUNS"
echo "Pacman wins: $pacman_wins"
echo "Ghost wins : $ghost_wins"

# Also append summary to logfile
echo "--" >> "$LOGFILE"
echo "Summary after $RUNS games:" >> "$LOGFILE"
echo "Pacman wins : $pacman_wins" >> "$LOGFILE"
echo "Ghost wins  : $ghost_wins" >> "$LOGFILE"
echo "Draws       : $draws" >> "$LOGFILE"
echo "Errors      : $errors" >> "$LOGFILE"
echo "Run finished: $(date)" >> "$LOGFILE"