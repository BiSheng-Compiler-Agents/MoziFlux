import importlib.util
from pathlib import Path

import triton


_ROUND4_PATH = Path(__file__).resolve().parents[1] / "opt-round-4" / "opt_34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"
_SPEC = importlib.util.spec_from_file_location("opt_round_4_module", _ROUND4_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load round-4 operator from {_ROUND4_PATH}")
_ROUND4_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ROUND4_MODULE)

ModelNew = _ROUND4_MODULE.ModelNew
get_inputs = _ROUND4_MODULE.get_inputs
get_init_inputs = _ROUND4_MODULE.get_init_inputs
_layernorm_gelu_scale_kernel = _ROUND4_MODULE._layernorm_gelu_scale_kernel


def _kernel_continuity_anchor(xc, y_out, w, b, n_rows, n_cols, eps, scale):
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_CTA"]),)
    _layernorm_gelu_scale_kernel[grid](
        xc,
        y_out,
        w,
        b,
        n_rows,
        n_cols,
        1.0 / float(n_cols),
        float(eps),
        float(scale),
    )
