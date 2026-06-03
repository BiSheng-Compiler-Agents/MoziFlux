#!/bin/bash
# run_kernel.sh — Build-and-run wrapper for cannsim_local_run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[RUNNER] Starting build+run in: $SCRIPT_DIR"

# Source CANN environment — required for both compile (torch_npu/ASCEND_HOME_PATH)
# and cmake (include paths for runtime/rt.h). CANNSIM_SETENV_PATH is set by the
# cannsim-local plugin; fall back to the standard CANN path if not set.
if [ -n "${CANNSIM_SETENV_PATH:-}" ] && [ -f "$CANNSIM_SETENV_PATH" ]; then
    echo "[RUNNER] Sourcing CANN env from CANNSIM_SETENV_PATH: $CANNSIM_SETENV_PATH"
    source "$CANNSIM_SETENV_PATH"
elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]; then
    echo "[RUNNER] Sourcing CANN env from ASCEND_HOME_PATH: $ASCEND_HOME_PATH/bin/setenv.bash"
    source "$ASCEND_HOME_PATH/bin/setenv.bash"
else
    echo "[RUNNER] WARNING: No CANN env found — compile or cmake may fail"
fi

# Step 1: Compile the kernel (ttir -> ttadapter -> npubin)
echo "[RUNNER] Compiling kernel..."
python vector_add.py
echo "[RUNNER] Kernel compiled OK"

# Step 2: Verify npubin exists
if [ ! -f "cannsim_host/add_kernel.npubin" ]; then
    echo "[RUNNER] ERROR: npubin not found!"
    exit 1
fi
echo "[RUNNER] npubin size: $(stat -c%s cannsim_host/add_kernel.npubin) bytes"

# Step 3: Build C++ host
echo "[RUNNER] Building C++ host..."
cd cannsim_host
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
make -j$(nproc) 2>&1
cp test_vector_add "$SCRIPT_DIR/"
cd "$SCRIPT_DIR"
echo "[RUNNER] Build OK, binary at: $SCRIPT_DIR/test_vector_add"

# Step 4: Copy npubin to same dir as binary
cp cannsim_host/add_kernel.npubin "$SCRIPT_DIR/add_kernel.npubin"

# Step 5: Done (cannsim will wrap this binary)
echo "[RUNNER] Ready for cannsim"
