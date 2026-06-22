# Optimizations

## 1. Replace custom scalar/vector depthwise convolution with ACL grouped Conv2d

**Before** (`82_conv_depthwise_2D_square_input_square_kernel.py`):
```python
_dwconv2d_kernel[grid](x, w, ..., BLOCK_W=256)
```
The baseline launches one Triton program per `(N, C, H_OUT, W_OUT tile)` and performs the 3x3 convolution as nine scalar/vector multiply-add loads.

**After** (`opt_82_conv_depthwise_2D_square_input_square_kernel.py`):
```python
return F.conv2d(
    x, self.conv2d.weight, self.conv2d.bias,
    stride=self.conv2d.stride,
    padding=self.conv2d.padding,
    dilation=self.conv2d.dilation,
    groups=self.conv2d.groups,
)
```
Rationale: depthwise Conv2d is a mature ACL-covered primitive; dispatching to ACL preserves the constructor, parameter ownership, grouping semantics, stride/padding, and initialization order while removing a structurally inefficient custom Triton launch.

## 2. Remove launch-grid overflow risk

The baseline target shape launches `N*C*H_OUT*ceil(W_OUT/256) = 16*64*510*2 = 1,044,480` Triton programs, far above Ascend's 65,535 launch-product limit. The optimized path has no custom Triton grid, so the overflow-prone dispatch is eliminated and scheduling is delegated to ACL.

## 3. Preserve public KernelBench interface

```python
def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    return [x]

def get_init_inputs():
    return [64, 3, 1, 0]
```
The optimized file keeps `ModelNew`, defaults, `get_inputs()`, and `get_init_inputs()` compatible with the editable baseline.
