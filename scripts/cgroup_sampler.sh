#!/bin/sh
# cgroup-v2 sampler for a running OpenClaw/OpenShell container.
# The first three columns retain the existing profiler schema. The remaining
# counters distinguish actual CPU use from cgroup quota throttling, and the
# pressure columns record PSI stall time so quota and contention are measurable.
set -eu

interval="${CGROUP_SAMPLE_INTERVAL:-0.100}"
trap 'exit 0' INT TERM EXIT

if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max
else
  cpu_quota=max
  cpu_period=100000
fi
printf '%s\n' 'epoch_ns,cpu_usage_usec,memory_bytes,user_usec,system_usec,nr_throttled,throttled_usec,cpu_quota,cpu_period,cpu_pressure_some_avg10,cpu_pressure_some_total,cpu_pressure_full_avg10,cpu_pressure_full_total,memory_pressure_some_avg10,memory_pressure_full_avg10,io_pressure_some_avg10,memory_inactive_file_bytes,memory_working_set_bytes,memory_anon_bytes,memory_file_bytes'

psi_field() {
  # $1=file $2=some|full $3=avg10|total
  if [ -r "$1" ]; then
    while IFS= read -r line; do
      case "$line" in
        "$2 "*)
          for token in $line; do
            case "$token" in
              "$3"=*) printf '%s' "${token#*=}"; return ;;
            esac
          done ;;
      esac
    done < "$1"
  fi
  printf '0'
}

while :; do
  cpu=0
  user=0
  system=0
  nr_throttled=0
  throttled=0
  if [ -r /sys/fs/cgroup/cpu.stat ]; then
    while read -r key value rest; do
      case "$key" in
        usage_usec) cpu="$value" ;;
        user_usec) user="$value" ;;
        system_usec) system="$value" ;;
        nr_throttled) nr_throttled="$value" ;;
        throttled_usec) throttled="$value" ;;
      esac
    done < /sys/fs/cgroup/cpu.stat
  fi
  if [ -r /sys/fs/cgroup/memory.current ]; then
    IFS= read -r memory < /sys/fs/cgroup/memory.current || true
  else
    memory=0
  fi
  inactive_file=0
  memory_anon=0
  memory_file=0
  if [ -r /sys/fs/cgroup/memory.stat ]; then
    while read -r key value rest; do
      case "$key" in
        inactive_file) inactive_file="$value" ;;
        anon) memory_anon="$value" ;;
        file) memory_file="$value" ;;
      esac
    done < /sys/fs/cgroup/memory.stat
  fi
  working_set=$((memory > inactive_file ? memory - inactive_file : 0))
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date +%s%N)" "$cpu" "$memory" "$user" "$system" "$nr_throttled" "$throttled" \
    "$cpu_quota" "$cpu_period" \
    "$(psi_field /sys/fs/cgroup/cpu.pressure some avg10)" \
    "$(psi_field /sys/fs/cgroup/cpu.pressure some total)" \
    "$(psi_field /sys/fs/cgroup/cpu.pressure full avg10)" \
    "$(psi_field /sys/fs/cgroup/cpu.pressure full total)" \
    "$(psi_field /sys/fs/cgroup/memory.pressure some avg10)" \
    "$(psi_field /sys/fs/cgroup/memory.pressure full avg10)" \
    "$(psi_field /sys/fs/cgroup/io.pressure some avg10)" \
    "$inactive_file" "$working_set" "$memory_anon" "$memory_file"
  sleep "$interval"
done
