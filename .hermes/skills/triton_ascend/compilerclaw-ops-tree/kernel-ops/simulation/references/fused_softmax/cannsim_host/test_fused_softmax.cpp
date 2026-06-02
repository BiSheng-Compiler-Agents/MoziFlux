#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>

// Only RT APIs — libruntime.so is swapped by cannsim for libruntime_camodel.so
// Do NOT include acl/acl.h or link libascendcl.so
#include "runtime/rt.h"

#define M           128
#define N           1024
#define BLOCK_N     1024
#define KERNEL_NAME "fused_softmax_kernel"

#define CHECK_RT(ret, msg) \
    do { if ((ret) != RT_ERROR_NONE) { \
        printf(msg " ERROR: 0x%x\n", (ret)); return (int)(ret); \
    } } while(0)

// Resolve kernel binary path relative to the executable
static std::string getKernelBinPath(const char* argv0) {
    std::string exe(argv0);
    auto pos = exe.find_last_of("/\\");
    std::string dir = (pos == std::string::npos) ? "." : exe.substr(0, pos);
    return dir + "/fused_softmax.npubin";
}

// Read binary file into a buffer
static std::vector<char> readBinary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        printf("Cannot open kernel binary: %s\n", path.c_str());
        return {};
    }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

// Packed args struct matching triton-ascend driver layout:
//   [syncBlockLock, workspace_addr, <kernel args...>, gridX, gridY, gridZ]
struct __attribute__((packed)) KernelArgs {
    void*   syncBlockLock;   // null
    void*   workspace_addr;  // null
    // kernel args (pointers first, then scalars, matching fn signature order)
    void*   x_ptr;
    void*   out_ptr;
    int32_t M_val;
    int32_t N_val;
    int32_t stride_xm;
    int32_t stride_om;
    // grid dims
    int32_t gridX;
    int32_t gridY;
    int32_t gridZ;
};

// Reference softmax on host for verification
static void ref_softmax(const std::vector<float>& x, std::vector<float>& out,
                        int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = -1e38f;
        for (int c = 0; c < cols; c++) mx = std::max(mx, x[r * cols + c]);
        float sum = 0.f;
        for (int c = 0; c < cols; c++) sum += std::exp(x[r * cols + c] - mx);
        for (int c = 0; c < cols; c++)
            out[r * cols + c] = std::exp(x[r * cols + c] - mx) / sum;
    }
}

int main(int argc, char* argv[]) {
    std::string kernelBinPath = getKernelBinPath(argc > 0 ? argv[0] : ".");

    // 1. Init device
    int32_t deviceId = 0;
    CHECK_RT(rtSetDevice(deviceId), "rtSetDevice failed.");
    printf("[INFO] Device set\n");

    rtStream_t stream;
    CHECK_RT(rtStreamCreate(&stream, 0), "rtStreamCreate failed.");
    printf("[INFO] Stream created\n");

    // 2. Load and register kernel binary
    auto kernelData = readBinary(kernelBinPath);
    if (kernelData.empty()) return -1;

    rtDevBinary_t devBin;
    devBin.magic   = RT_DEV_BINARY_MAGIC_ELF_AIVEC;   // AIV / vector kernel
    devBin.version = 0;
    devBin.data    = kernelData.data();
    devBin.length  = kernelData.size();

    void* binHandle = nullptr;
    CHECK_RT(rtDevBinaryRegister(&devBin, &binHandle), "rtDevBinaryRegister failed.");

    static size_t funcStub = 0;
    CHECK_RT(rtFunctionRegister(binHandle, &funcStub, KERNEL_NAME,
                                (void*)KERNEL_NAME, 0),
             "rtFunctionRegister failed.");
    printf("[INFO] Kernel binary registered\n");

    // 3. Prepare host data
    size_t numElem = (size_t)M * N;
    size_t dataSize = numElem * sizeof(float);

    std::vector<float> xHost(numElem), outHost(numElem, 0.f), refOut(numElem);

    // Fill with a pattern that produces clearly nonzero softmax outputs.
    // Use small values spread across [-2, 2] so softmax is not sharply peaked.
    // row r, col c → sin(r + c * 2*pi/N) — values in [-1, 1], uniform spread.
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            xHost[r * N + c] = std::sin(static_cast<float>(r) + c * 6.2831853f / N);

    ref_softmax(xHost, refOut, M, N);

    // 4. Allocate device memory
    void* xDev   = nullptr;
    void* outDev = nullptr;
    CHECK_RT(rtMalloc(&xDev,   dataSize, RT_MEMORY_HBM, 0), "rtMalloc x failed.");
    CHECK_RT(rtMalloc(&outDev, dataSize, RT_MEMORY_HBM, 0), "rtMalloc out failed.");

    CHECK_RT(rtMemcpy(xDev, dataSize, xHost.data(), dataSize,
                      RT_MEMCPY_HOST_TO_DEVICE), "rtMemcpy x H2D failed.");
    printf("[INFO] Device memory allocated (%d rows x %d cols)\n", M, N);

    // 5. Launch kernel — one block per row
    int gridX = M;
    int gridY = 1;
    int gridZ = 1;
    uint32_t blockNum = (uint32_t)(gridX * gridY * gridZ);

    KernelArgs args;
    args.syncBlockLock  = nullptr;
    args.workspace_addr = nullptr;
    args.x_ptr     = xDev;
    args.out_ptr   = outDev;
    args.M_val     = M;
    args.N_val     = N;
    args.stride_xm = N;
    args.stride_om = N;
    args.gridX     = gridX;
    args.gridY     = gridY;
    args.gridZ     = gridZ;

    CHECK_RT(rtKernelLaunch(&funcStub, blockNum,
                            static_cast<void*>(&args), sizeof(args),
                            nullptr, stream),
             "rtKernelLaunch failed.");

    // 6. Sync
    CHECK_RT(rtStreamSynchronize(stream), "rtStreamSynchronize failed.");
    printf("[INFO] Kernel launched and synced (blockNum=%u, %d rows)\n",
           blockNum, M);

    // 7. Copy back and verify
    CHECK_RT(rtMemcpy(outHost.data(), dataSize, outDev, dataSize,
                      RT_MEMCPY_DEVICE_TO_HOST), "rtMemcpy D2H failed.");

    bool correct = true;
    int  failures = 0;
    for (size_t i = 0; i < numElem && failures < 5; i++) {
        float diff = std::fabs(outHost[i] - refOut[i]);
        if (diff > 1e-4f) {
            printf("[FAIL] out[%zu] = %f  ref = %f  diff = %e\n",
                   i, outHost[i], refOut[i], diff);
            correct = false;
            failures++;
        }
    }

    // Print first row as sanity check — values should be clearly nonzero
    printf("[INFO] out[0,:8] = ");
    for (int c = 0; c < 8; c++) printf("%.6f ", outHost[c]);
    printf("...\n");
    printf("[INFO] ref[0,:8] = ");
    for (int c = 0; c < 8; c++) printf("%.6f ", refOut[c]);
    printf("...\n");

    // Sanity: each row must sum to 1.0
    bool sum_ok = true;
    for (int r = 0; r < M && sum_ok; r++) {
        float s = 0.f;
        for (int c = 0; c < N; c++) s += outHost[r * N + c];
        if (std::fabs(s - 1.0f) > 1e-3f) {
            printf("[FAIL] row %d sums to %f (expected 1.0)\n", r, s);
            correct = false; sum_ok = false;
        }
    }
    if (sum_ok) printf("[INFO] All rows sum to 1.0 (tol=1e-3)\n");

    if (correct)
        printf("[PASS] All %zu elements match reference (tol=1e-4)\n", numElem);

    // 8. Cleanup
    rtFree(xDev);
    rtFree(outDev);
    rtStreamDestroy(stream);
    rtDeviceReset(deviceId);
    return correct ? 0 : 1;
}
