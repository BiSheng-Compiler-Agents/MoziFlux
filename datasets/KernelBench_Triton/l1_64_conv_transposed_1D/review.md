# Static Review — opt_64_conv_transposed_1D.py

## P0

None found in the fp32 benchmark dispatch path: it preserves the source module weights and returns `nn.ConvTranspose1d(x.contiguous())` on NPU fp32 inputs.

## P1

- The fp16/bfloat16 Triton `tl.dot` path is cannsim-compiled, but remote hardware correctness did not complete because `remote_verify` timed out.
- The optimized low-precision path keeps the source restriction to stride=1, padding=0, output_padding=0, groups=1; fp32 dispatch is more general because it uses the vendor operator.

## P2

- `BLOCK_T=16`, `BLOCK_O=16`, `BLOCK_C=64` are conservative and should be autotuned on hardware if low-precision performance becomes the target.
- The weight tile access has stride `K` across output channels because PyTorch stores ConvTranspose1d weights as `[C_IN, C_OUT, K]`; a prepacked weight layout would improve MTE efficiency but would change module state handling.
