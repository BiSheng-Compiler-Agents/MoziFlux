# Cannsim C++ Host Build Pitfalls

## `_exit()` requires `<unistd.h>` on GCC 14+

`_exit(0)` is the recommended way to exit a cannsim host binary (it bypasses CANN 9.0.0's atexit segfault). On GCC 14+, `_exit` is not implicitly declared — it requires an explicit `#include <unistd.h>`:

```cpp
#include <unistd.h>   // _exit() — required on GCC 14+
```

**Symptom:**
```
error: '_exit' was not declared in this scope; did you mean '_Exit'?
```

**Fix:** Add `#include <unistd.h>` to the C++ host. The include is portable across all POSIX systems and GCC versions.

## `rtFunctionRegister` / `rtKernelLaunch` stub pattern on CANN 9.0.0+

Some CANN 9.0.0 runtime headers do not expose a usable `rtFunction_t` type for custom host launchers, and the `rtFunctionRegister` signature is:

```cpp
rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                   const void *kernelInfoExt, uint32_t funcMode);
```

Use a stable host-side stub object and launch with its address:

```cpp
static size_t func_stub = 0;
rtFunctionRegister(bin_handle, &func_stub, "my_kernel",
                   (void*)"my_kernel", 0);
rtKernelLaunch(&func_stub, GRID_X, &args, sizeof(args), nullptr, stream);
```

Avoid this incompatible pattern:

```cpp
rtFunction_t func = nullptr;
rtFunctionRegister(handle, (void*)"my_kernel", (void*)"my_kernel", 0, &func);
rtKernelLaunch(func, ...);
```

It can fail to compile with `rtFunction_t was not declared` or invalid argument conversions.

## `rtMemset` requires 4 arguments on CANN 9.0.0+

The CANN runtime `rtMemset` API has a 4-argument signature:

```cpp
rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t val, uint64_t cnt);
```

Passing only 3 arguments (as was valid on some earlier CANN versions) fails to compile on CANN 9.0.0:

```cpp
// ❌ WRONG — too few arguments on CANN 9.0.0
rtMemset(y_d, 0, buf_sz);

// ✅ Correct — pass all 4
rtMemset(y_d, buf_sz, 0, buf_sz);
```

**Symptom:**
```
error: too few arguments to function 'rtError_t rtMemset(void*, uint64_t, uint32_t, uint64_t)'
```

**Fix:** Always pass 4 arguments: `(ptr, destMax, val, cnt)`. Use `destMax = cnt` (the buffer size) as a conservative upper bound.

## `dirname()` requires `<libgen.h>` on GCC 14+

The C++ host example in the SKILL.md uses `dirname(argv[0])` to locate the `.npubin` file relative to the executable. On GCC 14+, `dirname` is not implicitly declared — it requires an explicit `#include <libgen.h>`:

```cpp
#include <libgen.h>   // dirname() — required on GCC 14+
```

**Symptom:** Build error at the line using `dirname`:
```
error: 'dirname' was not declared in this scope
```

**Fix:** Add `#include <libgen.h>` to the C++ host's includes. The include is portable across all GCC versions (11, 12, 13, 14+) and POSIX systems.

Note: on GCC < 12, `dirname` was implicitly declared via `<string>` or `<cstdlib>` transitive includes. GCC 14 removed the implicit declaration, making the missing include visible.

## `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` is required

`libruntime.so` has transitive dependencies (`liberror_manager.so`, `libmmpa.so`) that are not present at link time but ARE provided by the camodel at runtime. Without this flag:

**Symptom:** Linker errors about undefined symbols from `libruntime.so`.

**Fix:** Always pass `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` to cmake.

## Sub-kernel npubin pre-compilation pattern

When the Triton compile step inside `run_kernel.sh` fails (e.g., due to environment differences in the cannsim temp directory), you can pre-compile the `.npubin` separately and include it directly in `local_dir`:

1. Compile the kernel locally: `python3 compile_kernel.py` (works in the local workspace)
2. The `.npubin` is written to `local_dir/`
3. Have `run_kernel.sh` skip the compile step and only build the C++ host
4. `cannsim_local_run` copies everything to the temp dir, including the pre-compiled `.npubin`

This avoids CANN environment conflicts that can occur inside the cannsim temp directory.

## fp16 ↔ fp32 conversion in C++ host — `1 << (exp-15)` is UB for exp < 15

When writing a cannsim C++ host for elementwise kernels (e.g., tanh, sigmoid, activation functions), the host typically needs to convert between fp16 and fp32 for correctness checking. The naive approach:

```cpp
// ❌ WRONG — undefined behavior when exp < 15
float result = (1 + mant / 2048.0f) * (float)(1 << (exp - 15));
```

`1 << (exp - 15)` produces undefined behavior when `exp < 15` (the shift amount is negative). This happens for ALL fp16 values in the range (0, 1.0):

| fp16 value | exp field | exp-15 | Status |
|------------|-----------|--------|--------|
| 1.0 (0x3C00) | 15 | 0 | Safe |
| 0.5 (0x3800) | 14 | **-1** | **UB** |
| 0.1 (0x2E66) | 11 | **-4** | **UB** |
| tanh(-2.0)=-0.964 (0xBBF6) | 14 | **-1** | **UB** |
| subnormal (exp=0) | 0 | **-15** | **UB** |

**Symptom:** Output values appear as huge floats (~3×10⁹) because UB causes the shift to produce garbage bits. The host reports all checks as MISMATCH with `got=3.21074e+09`, even though the kernel actually computed the correct result.

**Fix:** Use IEEE 754 bit-level manipulation instead of arithmetic:

```cpp
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 31) {
        // NaN or Inf
        bits = sign | 0x7F800000 | (mant << 13);
    } else if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) { bits = sign; }
        else {
            exp = -1;
            uint32_t m = mant;
            while (!(m & 0x400)) { m <<= 1; exp--; }
            bits = sign | ((exp + 127 + 15) << 23) | ((m & 0x3FF) << 13);
        }
    } else {
        // Normal: shift the fp16 exponent and mantissa into fp32 positions
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t fp32_to_fp16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = (bits >> 13) & 0x3FF;
    uint32_t guard = (bits >> 12) & 1;       // round bit
    uint32_t sticky = (bits & 0xFFF) != 0 ? 1 : 0;  // sticky bit
    uint32_t round = guard & (mant | sticky);  // RNE rounding

    if (exp == 0xFF) {
        if (bits & 0x1FFF) return sign | 0x7E00 | (mant != 0);  // NaN
        return sign | 0x7C00;  // Inf
    }
    int32_t exp_unbiased = exp - 127;
    if (exp_unbiased < -14) return sign;        // underflow to zero
    if (exp_unbiased > 15) return sign | 0x7C00; // overflow to Inf

    return sign | ((exp_unbiased + 15) << 10) | (mant + round);
}
```

**Key points:**
- The fp16 → fp32 conversion places the 10-bit fp16 mantissa into bits [13:22] of the fp32 field, and adjusts the exponent bias (15 → 127).
- The fp32 → fp16 conversion truncates the 23-bit mantissa to 10 bits with round-to-nearest-even (RNE).
- Both conversions handle subnormal values (exp=0) and special values (NaN, Inf, zero).
- Verified on GCC 14.2.0, triton-ascend 3.2.0, CANN 9.0.0 (June 2026).
