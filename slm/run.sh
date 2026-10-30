#!/usr/bin/env bash
# job.sh under a deadline, its end marked in the log, then the VM released after a grace period
# for collecting results, so an unattended VM never keeps spending compute units.
cd "$(dirname "$0")"
timeout -k 120 "${DEADLINE_MINUTES:-300}m" bash job.sh
code=$?
if [ "$code" -eq 0 ]; then echo "== done"; else echo "== exit $code"; fi
nohup bash -c "sleep $((${GRACE_MINUTES:-25} * 60)); curl -s -X POST http://\$TBE_RUNTIME_ADDR/unassign" \
  >/dev/null 2>&1 &
