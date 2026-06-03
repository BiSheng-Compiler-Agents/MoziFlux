# Kernel Data Type Must Match C++ Host Buffers

## Problem

Passing fp32 (float) data to a kernel that expects fp16 causes silent data corruption
during simulation. The kernel reads 2 bytes at a time as fp16, but the host wrote 4-byte
fp32 values. The resulting fp16 values are completely different from the intended test data,
potentially causing incorrect reduction results (div by zero, inf/nan outputs).

## Symptom

- Simulation completes (419+ cycles for a log-softmax) but with `scalar_div: div by 0` errors
- Output values are random or zero after fp16-to-float conversion
- No compilation errors — the kernel runs, just with garbled data

## Fix

Match the kernel's declared dtype exactly:

```cpp
// Kernel signature: x_ptr: *fp16, y_ptr: *fp16
// Host must allocate fp16 (uint16_t) buffers

size_t in_bytes  = M * N * sizeof(uint16_t);  // 2 bytes per element
size_t out_bytes = M * N * sizeof(uint16_t);  // 2 bytes per element

uint16_t* h_in  = (uint16_t*)malloc(in_bytes);
uint16_t* h_out = (uint16_t*)malloc(out_bytes);

// Convert test data from float to fp16 before upload
float temp = 8.0f;
h_in[i] = float_to_fp16(temp);  // custom conversion function

// Upload fp16 data to device
rtMemcpy(xDev, in_bytes, h_in, in_bytes, RT_MEMCPY_HOST_TO_DEVICE);
```

## CANN Native fp16 Type

CANN defines a proper fp16 type in two headers:

| Header | Location | Content |
|---|---|---|
| `fp16.h` | `pkg_inc/op_common/op_host/util/` | Core `tagFp16` struct + `fp16_t` alias |
| `fp16_t.h` | `include/aclnn/opdev/` | Same struct + bit manipulation macros |

```cpp
// Core struct (both headers)
struct tagFp16 {
    uint16_t val;
    // constructors from uint16_t, float, double, int types
    // implicit conversion operators to float, double, uint16_t, etc.
    // arithmetic operators: +, -, *, /, +=, -=, *=, /=
};
using fp16_t = tagFp16;
```

`fp16_t` is a struct wrapping `uint16_t` — implicitly constructible from `uint16_t` and implicitly convertible back. For **raw buffer storage** in host launchers, `uint16_t` is correct and simplest. If you need a typed fp16 variable (e.g., for intermediate conversions), use `fp16_t`.

Bit layout is IEEE 754 half-precision: 1 sign bit, 5 exponent bits, 10 mantissa bits.

### Bit manipulation macros (fp16_t.h)

```cpp
#define FP16_SIGN_MASK  (0x8000)   // bit 15
#define FP16_EXP_MASK   (0x7C00)   // bits 14-10
#define FP16_MAN_MASK   (0x03FF)   // bits 9-0
#define FP16_EXP_BIAS   (15)
#define FP16_MAN_LEN    (10)

#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
#define FP16_EXTRAC_EXP(x)  (((x) >> 10) & 0x1F)
#define FP16_EXTRAC_MAN(x)  (((x) & 0x3FF) | ((((x) >> 10) & 0x1F) > 0 ? 0x400 : 0))
#define FP16_CONSTRUCTOR(s, e, m) \
    (static_cast<uint16_t>(((s) << 15) | ((e) << 10) | ((m) & 0x3FF)))
```

These macros are useful for manual bit-level fp16 manipulation in host launchers without relying on the struct's conversion operators.

## When This Applies

- **Always** when the kernel signature uses `*fp16`, `*bf16`, `*int8`, or any dtype
  that is not `*fp32` / `*float`
- Check the `*` type in the `compile(ASTSource(fn=..., signature={...}))` call
- The kernel's output type is also affected — if `y_ptr` is `*fp16`, the output
  buffer must be `uint16_t` sized, not `float` sized

## Conversion helpers

```cpp
// Float to fp16 (truncation)
inline uint16_t float_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = (u & 0x7FFFFF);
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 0xFF) {
        if (mant != 0) return (uint16_t)((sign << 15) | 0x7E00 | (mant >> 13));
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    int16_t new_exp = (int16_t)exp - 127 + 15;
    if (new_exp >= 31) return (uint16_t)((sign << 15) | 0x7C00);
    if (new_exp <= 0) return (uint16_t)(sign << 15);
    return ((uint16_t)sign << 15) | ((uint16_t)new_exp << 10) | (mant >> 13);
}

// fp16 to float
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | mant << 13;
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}
```
