#!/bin/sh
# Explicit operator-only installation. Never invoked by models or auto-deploy.
set -eu
[ "$(id -u)" = 0 ] || { echo 'Run as root'; exit 1; }
[ "${SPEC_PROVISION_DISK:-}" = 1 ] || { echo 'Set SPEC_PROVISION_DISK=1 after reviewing this script'; exit 1; }
root=/opt/spec_sandbox
image=/var/lib/spec-sandbox.img
[ ! -e "$image" ] || { echo 'Existing image: refusing overwrite'; exit 1; }
[ ! -e "$root" ] || [ -z "$(ls -A "$root")" ] || { echo 'Sandbox directory must be empty'; exit 1; }
available=$(df -PB1 /var/lib | awk 'NR==2 {print $4}')
[ "$available" -ge 40802189312 ] || { echo 'Require >=38GiB free to reserve 30GiB and retain >=8GiB'; exit 1; }
# Preallocated filesystem provides a hard disk ceiling; watchdog alone is not quota.
fallocate -l 30G "$image"
mkfs.ext4 -F -E nodiscard -m 0 "$image"
# Preserve allocation even on filesystems/toolchains that punch holes at mkfs.
fallocate -l 30G "$image"
mkdir -p "$root"
printf '%s %s ext4 loop,nodev,nosuid,X-fstrim.notrim 0 2\n' "$image" "$root" >> /etc/fstab
mount "$root"
# Automatic fstrim must not punch holes in the reservation. X-fstrim.notrim above
# persists across reboot; also disable discard on this exact attached loop now.
loopdev=$(findmnt -n -o SOURCE --target "$root")
case "$loopdev" in /dev/loop[0-9]*) ;; *) echo 'Unexpected sandbox backing device'; exit 1 ;; esac
loopname=$(basename "$loopdev")
[ "$(cat "/sys/block/$loopname/loop/backing_file")" = "$image" ] || { echo 'Backing file mismatch'; exit 1; }
printf '0\n' > "/sys/block/$loopname/queue/discard_max_bytes"
id spec-worker >/dev/null 2>&1 || useradd --system --home /nonexistent --shell /usr/sbin/nologin spec-worker
chown spec-worker:spec-worker "$root"
echo 'Disk pool provisioned. Install bubblewrap, Python, git and required runtimes; follow README before enabling worker.'
