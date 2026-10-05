#!/usr/bin/env bash
#
# enable-cgroup-memory.sh
#
# Raspberry Pi OS ships with the kernel's memory cgroup controller disabled
# by default, which makes Docker's --memory / mem_limit a silent no-op.
# This script checks for that, and if needed, enables it by appending
# cgroup_enable=memory cgroup_memory=1 to the boot cmdline, then (optionally)
# reboots.
#
# Usage:
#   sudo ./enable-cgroup-memory.sh            # interactive (asks before reboot)
#   sudo ./enable-cgroup-memory.sh -y          # auto-confirm the reboot
#   sudo ./enable-cgroup-memory.sh --no-reboot # edit cmdline.txt only, don't reboot
#
set -euo pipefail

AUTO_YES=0
DO_REBOOT=1

for arg in "$@"; do
  case "${arg}" in
    -y|--yes)
      AUTO_YES=1
      ;;
    --no-reboot)
      DO_REBOOT=0
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "Must be run as root (it edits a boot config file). Try: sudo $0 $*" >&2
  exit 1
fi

# --- 1. Is Docker already reporting memory limit support? -------------------
if command -v docker >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -qi "WARNING.*[Mm]emory"; then
    echo "== Docker reports missing memory limit support — cgroup fix needed =="
  else
    echo "== Docker already reports memory limit support enabled. Nothing to do. =="
    exit 0
  fi
else
  echo "== 'docker' not found on PATH; proceeding on the cmdline.txt check alone. =="
fi

# --- 2. Locate cmdline.txt (path differs across Raspberry Pi OS releases) ---
CMDLINE_FILE=""
for candidate in /boot/firmware/cmdline.txt /boot/cmdline.txt; do
  if [[ -f "${candidate}" ]]; then
    CMDLINE_FILE="${candidate}"
    break
  fi
done

if [[ -z "${CMDLINE_FILE}" ]]; then
  echo "Could not find cmdline.txt at /boot/firmware/cmdline.txt or /boot/cmdline.txt." >&2
  echo "This script only supports Raspberry Pi OS's standard boot layout." >&2
  exit 1
fi

echo "== Using ${CMDLINE_FILE} =="

# --- 3. Already patched? -----------------------------------------------------
if grep -qw "cgroup_enable=memory" "${CMDLINE_FILE}" && grep -qw "cgroup_memory=1" "${CMDLINE_FILE}"; then
  echo "== ${CMDLINE_FILE} already has cgroup_enable=memory cgroup_memory=1 =="
  echo "== If Docker still warns about missing memory support, reboot is likely still pending. =="
  exit 0
fi

# --- 4. Back up, then append in place (cmdline.txt must stay a single line) -
BACKUP_FILE="${CMDLINE_FILE}.bak.$(date +%Y%m%d%H%M%S)"
cp "${CMDLINE_FILE}" "${BACKUP_FILE}"
echo "== Backed up ${CMDLINE_FILE} to ${BACKUP_FILE} =="

# Strip the trailing newline (if any), append params with a single space,
# then write back a single line with exactly one trailing newline.
current="$(tr -d '\n' < "${CMDLINE_FILE}")"
printf '%s cgroup_enable=memory cgroup_memory=1\n' "${current}" > "${CMDLINE_FILE}"

echo "== Updated ${CMDLINE_FILE}: =="
cat "${CMDLINE_FILE}"
echo

# --- 5. Reboot (required for the new kernel cmdline to take effect) --------
if [[ "${DO_REBOOT}" -eq 0 ]]; then
  echo "== --no-reboot passed: change is written but NOT active until you reboot. =="
  exit 0
fi

if [[ "${AUTO_YES}" -eq 1 ]]; then
  echo "== Rebooting now (-y passed) =="
  reboot
else
  read -r -p "Reboot now to apply the change? [y/N] " reply
  case "${reply}" in
    [Yy]*)
      echo "== Rebooting =="
      reboot
      ;;
    *)
      echo "== Not rebooting. Change will take effect on your next reboot. =="
      ;;
  esac
fi
