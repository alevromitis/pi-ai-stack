#!/usr/bin/env bash
# Builds a patched liblitert-lm.so for the litert-lm image, on the Pi (arm64).
#
# The image installs litert-lm==0.18.0 from PyPI and then replaces the wheel's
# library with lib/out/liblitert-lm.so, built here from the v0.18.0 tag plus
# litert-lm-v0.18.0-mtp.patch (speculative decoding fixes; see the patch's
# header). Same Bazel target and flags as upstream's linux_arm64 wheel CI,
# YNNPACK included. The PyPI library has everything built in; this build
# (like upstream's) links two libraries dynamically, so lib/out/ gets three:
#   liblitert-lm.so                    the patched engine
#   libLiteRt.so                       LiteRT runtime with XNNPACK and YNNPACK,
#                                      built here from source
#   libGemmaModelConstraintProvider.so Google's closed-source constraint
#                                      provider, from the repo's LFS prebuilts
# liblitert-lm.so finds the other two next to itself (runpath $ORIGIN).
#
# Takes ~1.5 h on the Pi 5 with the other containers stopped (3 jobs, about
# 3 GB), ~3 h next to them (default CPU weight, so litert-lm-1 keeps the CPU).
# Work tree and Bazel cache: $LITERT_BUILD_DIR (default ~/litert-build,
# ~15 GB). Re-runs reuse the cache.
set -euo pipefail

[ "$(uname -m)" = aarch64 ] || { echo "build.sh: run this on the Pi (arm64)" >&2; exit 1; }
LIB="$(cd "$(dirname "$0")" && pwd)"
WORK="${LITERT_BUILD_DIR:-$HOME/litert-build}"
SRC="$WORK/LiteRT-LM-v0.18.0"
mkdir -p "$SRC" "$WORK/cache" "$WORK/home" "$LIB/out"

docker build -q -t litert-build-env:trixie -f "$LIB/Dockerfile.build-env" "$LIB" >/dev/null

# Everything git runs in the container: the clone uses Git LFS, which the
# host may not have. The tree is always /src in there: Bazel keys its cache
# on that path.
docker rm -f litert-build >/dev/null 2>&1 || true
docker run --rm --name litert-build --user 1000:1000 \
  -e HOME=/home/builder -e USER=builder \
  --memory 5g --memory-swap 9g \
  -v "$SRC:/src" -v "$WORK/cache:/cache" -v "$WORK/home:/home/builder" \
  -v "$LIB:/lib-src:ro" -v "$LIB/out:/out" \
  litert-build-env:trixie bash -euc '
    tag=v0.18.0
    [ -d /src/.git ] || git clone -q --filter=blob:none --branch "$tag" \
      https://github.com/google-ai-edge/LiteRT-LM.git /src
    cd /src
    git checkout -q --force "$tag"
    git clean -q -fdx -e "bazel-*"
    git apply /lib-src/litert-lm-v0.18.0-mtp.patch
    git lfs install --local >/dev/null
    git lfs pull --include="prebuilt/linux_arm64/*"
    bazel --output_user_root=/cache --host_jvm_args=-Xmx1200m build \
      --jobs=3 --local_ram_resources=3500 --local_cpu_resources=3 \
      --define=litert_runtime_link_mode=dynamic \
      --define=litert_enable_ynnpack=true \
      --config=linux_arm64 \
      -c opt //c:litert-lm
    # bazel-bin points into /cache, which only exists in here.
    install -m 0755 bazel-bin/c/liblitert-lm.so \
      bazel-bin/external/litert/litert/c/libLiteRt.so \
      prebuilt/linux_arm64/libGemmaModelConstraintProvider.so /out/
  '
(cd "$LIB/out" && sha256sum *.so > SHA256SUMS && cat SHA256SUMS)
