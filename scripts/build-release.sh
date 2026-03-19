#!/usr/bin/env bash
set -euo pipefail

# 建置 release binary（haro + haro-tui）並打包容器映像
# 用法: ./scripts/build-release.sh [--target TARGET] [--image]
#
# --target TARGET  指定 Rust target（預設偵測 host arch）
# --image          建置完後自動打包 haro-release 容器映像

TARGET=""
BUILD_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift 2 ;;
        --image) BUILD_IMAGE=true; shift ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

# 偵測 host arch → Rust target
detect_target() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64)  echo "x86_64-unknown-linux-gnu" ;;
        aarch64) echo "aarch64-unknown-linux-gnu" ;;
        armv7l)  echo "armv7-unknown-linux-gnueabihf" ;;
        *)       echo "" ;;
    esac
}

if [[ -z "$TARGET" ]]; then
    TARGET=$(detect_target)
fi

if [[ -n "$TARGET" ]]; then
    CARGO_ARGS=(--release --target "$TARGET")
    OUT_DIR="target/${TARGET}/release"
else
    CARGO_ARGS=(--release)
    OUT_DIR="target/release"
fi

echo "==> arch: $(uname -m), target: ${TARGET:-native}"
echo "==> 建置 release binary..."
cargo build "${CARGO_ARGS[@]}"

echo ""
echo "==> 建置完成："
for bin in haro haro-tui; do
    if [[ -f "${OUT_DIR}/${bin}" ]]; then
        SIZE=$(du -h "${OUT_DIR}/${bin}" | cut -f1)
        echo "    ${OUT_DIR}/${bin}  (${SIZE})"
    fi
done

if [[ "$BUILD_IMAGE" == true ]]; then
    echo ""
    echo "==> 建置容器映像 haro-release..."
    podman build \
        --build-arg "BIN_PATH=${OUT_DIR}/haro" \
        -t haro-release \
        -f deploy/Containerfile.haro .
    echo "==> 映像建置完成: localhost/haro-release:latest"
fi
