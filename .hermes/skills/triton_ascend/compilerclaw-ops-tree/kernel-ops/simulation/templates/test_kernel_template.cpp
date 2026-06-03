#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Only RT APIs — libruntime.so is swapped by cannsim for libruntime_camodel.so
// No libascendcl.so dependency (aclInit hits the real driver, not the
// simulator)
#include "runtime/rt.h"

#define M 128
#define N 128
#define K 64 /* 1×BLOCK_K — single iteration to minimize cannsim time */
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64
#define KERNEL_NAME "_matmul_kernel"
#define NPUBIN_NAME "my_kernel.npubin"

#define CHECK_RT(ret, msg)                                                     \
  do {                                                                         \
    if ((ret) != RT_ERROR_NONE) {                                              \
      printf(msg " ERROR: 0x%x\n", (ret));                                     \
      return (int)(ret);                                                       \
    }                                                                          \
  } while (0)

// Resolve kernel binary path relative to the executable.
// run_kernel.sh must copy NPUBIN_NAME next to the host binary.
static std::string getKernelBinPath(const char *argv0) {
  std::string exe(argv0);
  auto pos = exe.find_last_of("/\\");
  std::string dir = (pos == std::string::npos) ? "." : exe.substr(0, pos);
  return dir + "/" NPUBIN_NAME;
}

static std::vector<char> readBinary(const std::string &path) {
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
//   syncBlockLock, workspace_addr, kernel args in @triton.jit signature order,
//   gridX/Y/Z.
struct __attribute__((packed)) KernelArgs {
  void *syncBlockLock;
  void *workspace_addr;
  void *a_ptr;
  void *b_ptr;
  void *c_ptr;
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t stride_am;
  int32_t stride_ak;
  int32_t stride_bk;
  int32_t stride_bn;
  int32_t stride_cm;
  int32_t stride_cn;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};

int main(int argc, char *argv[]) {
  std::string kernelBinPath = getKernelBinPath(argc > 0 ? argv[0] : ".");

  int32_t deviceId = 0;
  CHECK_RT(rtSetDevice(deviceId), "rtSetDevice failed.");

  rtStream_t stream;
  CHECK_RT(rtStreamCreate(&stream, 0), "rtStreamCreate failed.");

  auto kernelData = readBinary(kernelBinPath);
  if (kernelData.empty())
    return -1;

  rtDevBinary_t devBin;
  devBin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  devBin.version = 0;
  devBin.data = kernelData.data();
  devBin.length = kernelData.size();

  void *binHandle = nullptr;
  CHECK_RT(rtDevBinaryRegister(&devBin, &binHandle),
           "rtDevBinaryRegister failed.");

  static size_t funcStub = 0;
  CHECK_RT(rtFunctionRegister(binHandle, &funcStub, KERNEL_NAME,
                              (void *)KERNEL_NAME, 0),
           "rtFunctionRegister failed.");

  int m = M, n = N, k = K;
  int stride_am = k, stride_ak = 1;
  int stride_bk = n, stride_bn = 1;
  int stride_cm = n, stride_cn = 1;
  int gridX = 1, gridY = 1, gridZ = 1;

  size_t sizeA = (size_t)m * k * sizeof(uint16_t);
  size_t sizeB = (size_t)k * n * sizeof(uint16_t);
  size_t sizeC = (size_t)m * n * sizeof(uint16_t);

  std::vector<uint16_t> aHost((size_t)m * k, 1);
  std::vector<uint16_t> bHost((size_t)k * n, 1);
  std::vector<uint16_t> cHost((size_t)m * n, 0);

  void *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
  CHECK_RT(rtMalloc(&aDev, sizeA, RT_MEMORY_HBM, 0), "rtMalloc a failed.");
  CHECK_RT(rtMalloc(&bDev, sizeB, RT_MEMORY_HBM, 0), "rtMalloc b failed.");
  CHECK_RT(rtMalloc(&cDev, sizeC, RT_MEMORY_HBM, 0), "rtMalloc c failed.");

  CHECK_RT(rtMemcpy(aDev, sizeA, aHost.data(), sizeA, RT_MEMCPY_HOST_TO_DEVICE),
           "rtMemcpy a H2D failed.");
  CHECK_RT(rtMemcpy(bDev, sizeB, bHost.data(), sizeB, RT_MEMCPY_HOST_TO_DEVICE),
           "rtMemcpy b H2D failed.");

  KernelArgs args;
  args.syncBlockLock = nullptr;
  args.workspace_addr = nullptr;
  args.a_ptr = aDev;
  args.b_ptr = bDev;
  args.c_ptr = cDev;
  args.m = m;
  args.n = n;
  args.k = k;
  args.stride_am = stride_am;
  args.stride_ak = stride_ak;
  args.stride_bk = stride_bk;
  args.stride_bn = stride_bn;
  args.stride_cm = stride_cm;
  args.stride_cn = stride_cn;
  args.gridX = gridX;
  args.gridY = gridY;
  args.gridZ = gridZ;

  uint32_t blockNum = (uint32_t)(gridX * gridY * gridZ);
  printf("[HOST] Launching kernel (grid=%u, blockNum=%u)...\n",
         gridX * gridY * gridZ, blockNum);
  CHECK_RT(rtKernelLaunch(&funcStub, blockNum, static_cast<void *>(&args),
                          sizeof(args), nullptr, stream),
           "rtKernelLaunch failed.\n");

  CHECK_RT(rtStreamSynchronize(stream), "rtStreamSynchronize failed.\n");
  printf("[HOST] Kernel completed\n");
  CHECK_RT(rtMemcpy(cHost.data(), sizeC, cDev, sizeC, RT_MEMCPY_DEVICE_TO_HOST),
           "rtMemcpy c D2H failed.\n");

  printf("[INFO] First 4 elements of C: ");
  for (int i = 0; i < 4 && i < m * n; i++)
    printf("%04x ", cHost[i]);
  printf("\n");

  // Correctness check: A and B are all 1.0 (bf16), so C[i] should be k * 1.0 =
  // k (in fp32). Convert bf16 output to fp32 and compare against expected
  // value.
  auto bf16_to_fp32 = [](uint16_t h) -> float {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  };

  float expected = (float)k;
  int check_count = std::min(m * n, 64);
  int fail_count = 0;
  for (int i = 0; i < check_count; i++) {
    float got = bf16_to_fp32(cHost[i]);
    float diff = std::fabs(got - expected);
    float tol = std::max(1.0f, std::fabs(expected)) * 0.01f; // 1% tolerance
    if (diff > tol) {
      if (fail_count < 4) {
        printf("[FAIL] C[%d] = %.2f, expected %.2f (diff=%.2f, tol=%.2f)\n", i,
               got, expected, diff, tol);
      }
      fail_count++;
    }
  }

  if (fail_count > 0) {
    printf("[FAIL] %d/%d elements failed correctness check\n", fail_count,
           check_count);
    rtFree(aDev);
    rtFree(bDev);
    rtFree(cDev);
    rtStreamDestroy(stream);
    rtDeviceReset(deviceId);
    return 1;
  }
  printf("[HOST] PASS — %d/%d elements correct (expected=%.1f)\n", check_count,
         check_count, expected);

  rtFree(aDev);
  rtFree(bDev);
  rtFree(cDev);
  rtStreamDestroy(stream);
  rtDeviceReset(deviceId);
  return 0;
}
