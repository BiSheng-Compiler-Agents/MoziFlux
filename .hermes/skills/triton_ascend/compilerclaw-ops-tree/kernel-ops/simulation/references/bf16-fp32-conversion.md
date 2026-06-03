# bf16 ↔ fp32 Conversion for Host Launcher Correctness Checks

## bf16 → fp32 (for reading kernel output)

```cpp
#include <cstdint>
#include <cstring>

float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
```

This is the canonical pattern — bit-shift the bf16 representation into the upper 16 bits of a uint32, then memcpy into float. No UB (unlike union-based punning).

## fp32 → bf16 (for preparing input buffers)

```cpp
#include <cmath>

uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    // Round-to-nearest-even: add 0x7fff + carry bit, then truncate
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fff + lsb;
    return (uint16_t)(bits >> 16);
}
```

## Usage in correctness checks

When the host initializes all inputs to 1.0 (bf16), the expected matmul output is `k * 1.0 = k` (fp32). Check with 1% relative tolerance:

```cpp
float expected = (float)k;
float got = bf16_to_fp32(cHost[i]);
float diff = std::fabs(got - expected);
float tol = std::max(1.0f, std::fabs(expected)) * 0.01f;
if (diff > tol) { /* fail */ }
```

## CANN Native bf16 Type

CANN defines a proper bf16 type in `bfloat16.h`:

```cpp
// From pkg_inc/op_common/op_host/util/bfloat16.h (also in include/aclnn/opdev/)
struct bfloat16 {
    uint16_t value;
    // constructors from uint16_t, float, double, int types
    // implicit conversion operators to float, double, etc.
    // arithmetic operators: +, -, *, /, +=, -=, *=, /=
    // static helpers: round_to_bfloat16(), epsilon(), highest(), lowest()
};
```

`bfloat16` is a struct wrapping `uint16_t` — implicitly constructible from `uint16_t` and implicitly convertible back. For **raw buffer storage** in host launchers, `uint16_t` is correct and simplest. If you need a typed bf16 variable (e.g., for intermediate conversions), use `bfloat16`.

Bit layout: 1 sign bit, 8 exponent bits (same as fp32), 7 mantissa bits.

**Note on fp16 vs bf16:** CANN also provides `fp16_t` (aka `tagFp16`) for IEEE 754 half-precision in `fp16.h` / `fp16_t.h`, with the same struct-wrapping-`uint16_t` pattern. The `fp16_t.h` header additionally provides bit manipulation macros (`FP16_SIGN_MASK`, `FP16_EXP_MASK`, `FP16_MAN_MASK`, `FP16_EXTRAC_*`, `FP16_CONSTRUCTOR`). See `references/cannsim-cpp-host-dtype-matching.md` for details.

## Pitfalls

- **Don't use `__bf16` or `_Float16` directly** — Ascend's bf16 is stored as `uint16_t` in host buffers. The kernel dtype `*bf16` maps to `uint16_t*` in C++, not `__bf16*`.
- **Don't cast `(float)bf16_val`** — that reinterprets the uint16 as an integer and converts to float (e.g., `0x3F80` → `16256.0f`), not the bf16 value.
