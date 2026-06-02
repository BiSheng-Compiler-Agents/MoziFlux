#!/bin/bash
# cannsim wrapper for vector_add
# Compiles the kernel + C++ host binary on the remote (if not already done).
# cannsim record wraps the binary directly — this script is the build step only.
# Uses $(dirname "$0") so paths are correct regardless of upload location.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 1: compile the Triton kernel → npubin (skipped if npubin already present)
if [ ! -f "$SCRIPT_DIR/cannsim_host/add_kernel.npubin" ]; then
    echo "[BUILD] Compiling add_kernel → npubin..."
    python "$SCRIPT_DIR/vector_add.py"
    echo "[BUILD] Kernel compile done."
fi

# Step 2: build the C++ host binary (skipped if already built)
if [ ! -f "$SCRIPT_DIR/test_vector_add" ]; then
    echo "[BUILD] Compiling test_vector_add on remote..."
    cd "$SCRIPT_DIR/cannsim_host"
    mkdir -p build && cd build
    cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_BUILD_TYPE=Release 2>&1
    make -j$(nproc) 2>&1
    cp bin/test_vector_add "$SCRIPT_DIR/test_vector_add"
    echo "[BUILD] Done."
fi

# Step 3: ensure the npubin sits next to the host binary (binary resolves via dirname(argv[0]))
if [ ! -f "$SCRIPT_DIR/add_kernel.npubin" ] && [ -f "$SCRIPT_DIR/cannsim_host/add_kernel.npubin" ]; then
    cp "$SCRIPT_DIR/cannsim_host/add_kernel.npubin" "$SCRIPT_DIR/add_kernel.npubin"
fi
