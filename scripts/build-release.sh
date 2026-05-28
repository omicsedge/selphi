#!/usr/bin/env bash
# Build release binaries for distribution.
#
# Default: glibc dynamic for the host architecture (fast, requires glibc 2.28+).
# --musl:  also build a fully static musl binary (portable to Alpine/scratch
#          Docker/ancient distros, but ~60% slower in the HMM hot path).
#
# Outputs land in dist/ with platform-suffixed names.
#
# Requirements:
#   glibc target: rustc, system gcc/clang
#   musl target:  apt install musl-tools; rustup target add x86_64-unknown-linux-musl
#                 (or aarch64-unknown-linux-musl on ARM)
#
# Usage:
#   ./scripts/build-release.sh                # glibc only
#   ./scripts/build-release.sh --musl         # glibc + musl
#   ./scripts/build-release.sh --musl-only    # musl only

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
DIST=$REPO_ROOT/dist
mkdir -p "$DIST"

WANT_GLIBC=1
WANT_MUSL=0
for arg in "$@"; do
  case "$arg" in
    --musl)       WANT_MUSL=1 ;;
    --musl-only)  WANT_MUSL=1; WANT_GLIBC=0 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Detect host architecture for output naming.
HOST_ARCH=$(uname -m)   # x86_64 or aarch64
OS=$(uname -s | tr 'A-Z' 'a-z')

build_glibc() {
  local triple
  case "$HOST_ARCH" in
    x86_64)  triple=x86_64-unknown-linux-gnu ;;
    aarch64) triple=aarch64-unknown-linux-gnu ;;
    *) echo "unsupported arch $HOST_ARCH" >&2; exit 1 ;;
  esac
  echo "=== building glibc dynamic for $triple ==="
  cargo build --release --target "$triple"
  local out="$DIST/selphi-${OS}-${HOST_ARCH}"
  cp "target/$triple/release/selphi" "$out"
  echo "→ $out  ($(stat -c%s "$out" | numfmt --to=iec))"
  ldd "$out" 2>&1 | sed 's/^/   /'
}

build_musl() {
  local triple
  case "$HOST_ARCH" in
    x86_64)  triple=x86_64-unknown-linux-musl ;;
    aarch64) triple=aarch64-unknown-linux-musl ;;
    *) echo "unsupported arch $HOST_ARCH" >&2; exit 1 ;;
  esac
  if ! rustup target list --installed | grep -q "^${triple}$"; then
    echo "ERROR: rustup target $triple not installed. Run: rustup target add $triple" >&2
    exit 1
  fi
  if ! command -v musl-gcc >/dev/null 2>&1; then
    echo "ERROR: musl-gcc not found. Run: sudo apt install musl-tools" >&2
    exit 1
  fi
  echo "=== building musl static for $triple ==="
  CC_x86_64_unknown_linux_musl=musl-gcc \
  AR_x86_64_unknown_linux_musl=ar \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
    cargo build --release --target "$triple"
  local out="$DIST/selphi-${OS}-${HOST_ARCH}-musl"
  cp "target/$triple/release/selphi" "$out"
  echo "→ $out  ($(stat -c%s "$out" | numfmt --to=iec))"
  file "$out" | sed 's/^/   /'
}

[ "$WANT_GLIBC" = 1 ] && build_glibc
[ "$WANT_MUSL"  = 1 ] && build_musl

echo
echo "=== built artifacts in $DIST ==="
ls -lh "$DIST"
