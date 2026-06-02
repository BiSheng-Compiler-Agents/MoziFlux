#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

// Only RT APIs — libruntime.so is swapped by cannsim for libruntime_camodel.so
// No libascendcl.so dependency (aclInit hits the real driver, not the simulator)
#include "runtime/rt.h"

#define N          1024
#define BLOCK_SIZE 1024
#define KERNEL_NAME "add_kernel"

#define CHECK_RT(ret, msg) \
    do { if ((ret) != RT_ERROR_NONE) { \
        printf(msg " ERROR: 0x%x\n", (ret)); return (int)(ret); \
    } } while(0)

// Resolve kernel binary path relative to the executable
static std::string getKernelBinPath(const char* argv0) {
    std::string exe(argv0);
    auto pos = exe.find_last_of("/\\");
    std::string dir = (pos == std::string::npos) ? "." : exe.substr(0, pos);
    return dir + "/add_kernel.npubin";
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

// Packed args struct matching triton-ascend driver layout
struct __attribute__((packed)) KernelArgs {
    void*   syncBlockLock;   // workspace lock (null)
    void*   workspace_addr;  // workspace ptr  (null)
    void*   x_ptr;
    void*   y_ptr;
    void*   out_ptr;
    int32_t n;
    int32_t gridX;
    int32_t gridY;
    int32_t gridZ;
};

int main(int argc, char* argv[]) {
    std::string kernelBinPath = getKernelBinPath(argc > 0 ? argv[0] : ".");

    // 1. Init device using RT APIs directly (cannsim intercepts these)
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
    devBin.magic   = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    devBin.version = 0;
    devBin.data    = kernelData.data();
    devBin.length  = kernelData.size();

    void* binHandle = nullptr;
    CHECK_RT(rtDevBinaryRegister(&devBin, &binHandle), "rtDevBinaryRegister failed.");

    static size_t funcStub = 0;
    CHECK_RT(rtFunctionRegister(binHandle, &funcStub, KERNEL_NAME, (void*)KERNEL_NAME, 0),
             "rtFunctionRegister failed.");
    printf("[INFO] Kernel binary registered\n");

    // 3. Allocate device memory using RT APIs
    size_t dataSize = N * sizeof(float);

    std::vector<float> xHost(N, 1.0f);
    std::vector<float> yHost(N, 2.0f);
    std::vector<float> outHost(N, 0.0f);

    void* xDev   = nullptr;
    void* yDev   = nullptr;
    void* outDev = nullptr;

    CHECK_RT(rtMalloc(&xDev,   dataSize, RT_MEMORY_HBM, 0), "rtMalloc x failed.");
    CHECK_RT(rtMalloc(&yDev,   dataSize, RT_MEMORY_HBM, 0), "rtMalloc y failed.");
    CHECK_RT(rtMalloc(&outDev, dataSize, RT_MEMORY_HBM, 0), "rtMalloc out failed.");

    CHECK_RT(rtMemcpy(xDev,   dataSize, xHost.data(),   dataSize, RT_MEMCPY_HOST_TO_DEVICE),
             "rtMemcpy x H2D failed.");
    CHECK_RT(rtMemcpy(yDev,   dataSize, yHost.data(),   dataSize, RT_MEMCPY_HOST_TO_DEVICE),
             "rtMemcpy y H2D failed.");

    printf("[INFO] Device memory allocated (x=1.0, y=2.0, expect out=3.0)\n");

    // 4. Launch kernel
    int gridX = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // = 1
    int gridY = 1;
    int gridZ = 1;
    uint32_t blockNum = (uint32_t)(gridX * gridY * gridZ);

    KernelArgs args;
    args.syncBlockLock  = nullptr;
    args.workspace_addr = nullptr;
    args.x_ptr   = xDev;
    args.y_ptr   = yDev;
    args.out_ptr = outDev;
    args.n       = N;
    args.gridX   = gridX;
    args.gridY   = gridY;
    args.gridZ   = gridZ;

    CHECK_RT(rtKernelLaunch(&funcStub, blockNum,
                            static_cast<void*>(&args), sizeof(args),
                            nullptr, stream),
             "rtKernelLaunch failed.");

    // 5. Synchronize
    CHECK_RT(rtStreamSynchronize(stream), "rtStreamSynchronize failed.");
    printf("[INFO] Kernel launched and synced (blockNum=%u)\n", blockNum);

    // 6. Copy results back
    CHECK_RT(rtMemcpy(outHost.data(), dataSize, outDev, dataSize, RT_MEMCPY_DEVICE_TO_HOST),
             "rtMemcpy D2H failed.");

    // 7. Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (outHost[i] != 3.0f) {
            printf("[FAIL] out[%d] = %f (expected 3.0)\n", i, outHost[i]);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("[PASS] All %d elements are 3.0\n", N);

    // 8. Cleanup
    rtFree(xDev);
    rtFree(yDev);
    rtFree(outDev);
    rtStreamDestroy(stream);
    rtDeviceReset(deviceId);
    return correct ? 0 : 1;
}
