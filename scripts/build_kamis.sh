#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAMIS_DIR="$ROOT_DIR/external/KaMIS"
MMWIS_DIR="$KAMIS_DIR/mmwis"
MMWIS_BINARY="$MMWIS_DIR/deploy/mmwis"

echo "Initializing KaMIS submodules..."
git -C "$ROOT_DIR" submodule update --init --recursive

case "$(uname -s)" in
    Darwin)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Homebrew is required on macOS to install GNU GCC."
            echo "Install Homebrew, then run this script again."
            exit 1
        fi

        if ! brew --prefix gcc >/dev/null 2>&1; then
            echo "Installing GNU GCC with Homebrew..."
            brew install gcc
        fi

        GCC_PREFIX="$(brew --prefix gcc)"
        GCC_BIN="$(find "$GCC_PREFIX/bin" -maxdepth 1 -type f -name 'gcc-[0-9]*' | sort | head -n 1)"
        GXX_BIN="$(find "$GCC_PREFIX/bin" -maxdepth 1 -type f -name 'g++-[0-9]*' | sort | head -n 1)"

        if [[ -z "$GCC_BIN" || -z "$GXX_BIN" ]]; then
            echo "Could not locate Homebrew GNU GCC executables."
            exit 1
        fi

        SHIM_DIR="$(mktemp -d)"
        trap 'rm -rf "$SHIM_DIR"' EXIT

        ln -s "$GCC_BIN" "$SHIM_DIR/gcc"
        ln -s "$GXX_BIN" "$SHIM_DIR/g++"

        echo "Building KaMIS MMWIS with GNU GCC..."
        (
            cd "$MMWIS_DIR"
            PATH="$SHIM_DIR:$PATH" ./compile.sh Release
        )
        ;;

    Linux)
        for command_name in cmake gcc g++ make; do
            if ! command -v "$command_name" >/dev/null 2>&1; then
                echo "Missing required build tool: $command_name"
                exit 1
            fi
        done

        echo "Building KaMIS MMWIS..."
        (
            cd "$MMWIS_DIR"
            ./compile.sh Release
        )
        ;;

    *)
        echo "Unsupported operating system: $(uname -s)"
        exit 1
        ;;
esac

if [[ ! -x "$MMWIS_BINARY" ]]; then
    echo "KaMIS build failed: mmwis executable was not created."
    exit 1
fi

echo
echo "KaMIS MMWIS built successfully:"
echo "$MMWIS_BINARY"
