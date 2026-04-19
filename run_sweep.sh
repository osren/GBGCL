#!/bin/bash
LOGFILE="sweepX_$(date +%Y%m%d_%H%M%S).log"
ERRFILE="sweepX_errors_$(date +%Y%m%d_%H%M%S).log"

echo "Log: $LOGFILE"
echo "Errors: $ERRFILE"

python tools/sweepX.py 2>&1 | tee "$LOGFILE" | grep -E "\[ERR " > "$ERRFILE"
