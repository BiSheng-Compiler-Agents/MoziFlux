# CANN RT function registration stub pattern

Some CANN runtime headers do not define `rtFunction_t`. For cannsim C++ hosts, register a Triton function with a `static size_t` stub and launch by pointer to that stub.

```cpp
static size_t func_stub = 0;
check(rtFunctionRegister(bin_handle,
                         &func_stub,
                         "_kernel_name",
                         (void*)"_kernel_name",
                         0),
      "rtFunctionRegister");

KernelArgs args{/* sync/workspace, kernel args, grid dims */};
check(rtKernelLaunch(&func_stub, gridX, &args, sizeof(args), nullptr, stream),
      "rtKernelLaunch");
```

Avoid this non-portable pattern, which fails on some CANN installs:

```cpp
rtFunction_t func = nullptr;                 // may be undefined
rtFunctionRegister(handle, ..., &func);
rtKernelLaunch(func, ...);
```

The function name must still match the Python `@triton.jit` def name exactly.
