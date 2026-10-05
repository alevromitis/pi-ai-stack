#!/usr/bin/env bash
#
# SOPS-encrypt / decrypt each stack's git-ignored env files.
#
#   scripts/secrets.sh encrypt [stack-dir...]
#   scripts/secrets.sh decrypt [stack-dir...]
#
# With no stack-dir, every directory under docker/prod and docker/dev that
# holds a docker-compose.*.yml. `make encrypt` / `make decrypt` in a stack
# dir (compose.mk) call this with that one dir.
#
# encrypt: every .env / .env.* (not *.example, *.sops) -> <name>.sops beside
#          it, dotenv format, recipient from the repo's .sops.yaml. Needs only
#          the public key. A .sops is rewritten only when its content would
#          change: compared by decrypting it when the private key is at hand,
#          otherwise by mtime (plaintext newer than .sops), so re-runs leave
#          git clean.
# decrypt: every *.sops -> its plaintext sibling, mode 600. Needs the private
#          key: $SOPS_AGE_KEY, or $SOPS_AGE_KEY_FILE (default
#          ~/.config/sops/age/keys.txt). An existing plaintext that differs is
#          left alone (reported, exit 1) unless FORCE=1.
#
# The dotenv format sops reads is strict: KEY=VALUE or `# comment` lines
# starting in column 1. Blank lines are dropped on the round trip; quotes
# are kept literally, as Compose's env_file reads them.

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/.sops.yaml"
export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"

die()  { printf 'secrets.sh: %s\n' "$*" >&2; exit 1; }
warn() { printf 'secrets.sh: %s\n' "$*" >&2; }

usage() {
  sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

# Half-written temp file of an interrupted run (it would match .env.*).
tmp=
trap 'rm -f "${tmp:-}"' EXIT

command -v sops >/dev/null || die "sops not found in PATH"
[ -f "$CONFIG" ] || die "missing $CONFIG"

have_key() {
  [ -n "${SOPS_AGE_KEY:-}" ] || [ -r "$SOPS_AGE_KEY_FILE" ]
}

# What sops' dotenv round trip yields for a plaintext file: blank lines
# dropped, newline-terminated.
normalise() { sed -e '/^[[:space:]]*$/d' -e '$a\' "$1"; }

sops_d() { sops --config "$CONFIG" decrypt --input-type dotenv --output-type dotenv "$1"; }

# Plaintext env files in a stack dir, one per line.
plain_files() {
  local f
  for f in "$1"/.env "$1"/.env.*; do
    [ -f "$f" ] || continue
    case "$f" in *.example|*.sops) continue ;; esac
    printf '%s\n' "$f"
  done
}

encrypt_dir() {
  local dir=$1 f out
  while IFS= read -r f; do
    out="$f.sops"
    if [ -e "$out" ]; then
      if have_key; then
        if cmp -s <(normalise "$f") <(sops_d "$out" 2>/dev/null); then
          printf '  same      %s\n' "${out#"$ROOT"/}"; continue
        fi
      elif [ ! "$f" -nt "$out" ]; then
        printf '  same      %s (mtime; no private key to compare)\n' "${out#"$ROOT"/}"; continue
      fi
    fi
    tmp=$(mktemp "$out.XXXXXX")
    if ! sops --config "$CONFIG" encrypt --input-type dotenv --output-type dotenv \
         --filename-override "$out" "$f" >"$tmp"; then
      rm -f "$tmp"
      warn "failed to encrypt $f (dotenv lines must be KEY=VALUE or '# comment' from column 1)"
      status=1; continue
    fi
    chmod 644 "$tmp"
    mv -f "$tmp" "$out"
    # Same mtime as the plaintext: the key-less mtime check sees it as done.
    touch -r "$f" "$out"
    printf '  encrypted %s\n' "${out#"$ROOT"/}"
  done < <(plain_files "$dir")
}

decrypt_dir() {
  local dir=$1 enc f
  for enc in "$dir"/.env.sops "$dir"/.env.*.sops; do
    [ -f "$enc" ] || continue
    f=${enc%.sops}
    tmp=$(umask 077 && mktemp "$f.XXXXXX")
    if ! sops_d "$enc" >"$tmp"; then
      rm -f "$tmp"; warn "failed to decrypt $enc"; status=1; continue
    fi
    if [ -e "$f" ]; then
      if cmp -s <(normalise "$f") "$tmp"; then
        rm -f "$tmp"; chmod 600 "$f"
        printf '  same      %s\n' "${f#"$ROOT"/}"; continue
      fi
      if [ "${FORCE:-0}" != 1 ]; then
        rm -f "$tmp"
        warn "$f differs from $enc; left alone (FORCE=1 overwrites, or 'make encrypt' to keep the local copy)"
        status=1; continue
      fi
    fi
    chmod 600 "$tmp"
    mv -f "$tmp" "$f"
    touch -r "$enc" "$f"
    printf '  decrypted %s\n' "${f#"$ROOT"/}"
  done
}

[ $# -ge 1 ] || usage
action=$1; shift
case "$action" in
  encrypt) ;;
  decrypt)
    have_key || die "decrypt needs the age private key and none was found at
  $SOPS_AGE_KEY_FILE (SOPS_AGE_KEY_FILE) and SOPS_AGE_KEY is unset.
  Copy it from the password manager, e.g.:
    mkdir -p \"$(dirname "$SOPS_AGE_KEY_FILE")\" && install -m 600 /dev/stdin \"$SOPS_AGE_KEY_FILE\"   # paste, then Ctrl-D
  or run with SOPS_AGE_KEY_FILE=/path/to/keys.txt. encrypt needs no private key." ;;
  *) usage ;;
esac

dirs=("$@")
if [ ${#dirs[@]} -eq 0 ]; then
  while IFS= read -r c; do dirs+=("$(dirname "$c")"); done < <(
    find "$ROOT/docker/prod" "$ROOT/docker/dev" -mindepth 2 -maxdepth 2 \
      -name 'docker-compose*.yml' 2>/dev/null | sort -u)
fi

status=0
for d in "${dirs[@]}"; do
  [ -d "$d" ] || die "not a directory: $d"
  "${action}_dir" "${d%/}"
done
exit "$status"
