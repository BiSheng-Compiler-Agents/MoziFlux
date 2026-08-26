import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
import triton.runtime.driver as driver

_CAUSAL_MASK_CACHE = {}


def _causal_mask(n_ctx, device):
    key = (device, n_ctx)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones((n_ctx, n_ctx), device=device,
                          dtype=torch.bool).triu(diagonal=1)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    ATTEN_MASK,
    sm_scale,
    TMP,
    L,
    M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUB_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Structure (user HW recipe + bisect-proven): the persistent tile loop stays
    # OUTSIDE the scopes — it does not touch the CV buffers' memory planning.
    # Only the start_n loop (which repeatedly uses qk_ub/p_l1/pv_ub) lives
    # INSIDE each scope. Per tile there is one cube scope + one vector scope;
    # the two sides communicate ONLY through bl.alloc buffers + sync_block
    # flags, and all tensor state is created/updated/stored within one scope.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_blocks_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_blocks_m * Z * H

    # UB buffers are per-vector-core halves (fixpipe ROW_SPLIT drains L0C rows
    # [0, M/2) -> AIV0 UB, [M/2, M) -> AIV1 UB). p_l1: P in NZ fractal for PV.
    qk_ub = bl.alloc(tl.float32, [SUB_M, BLOCK_N],
                     _address_space=al.ascend_address_space.UB)
    p_l1 = bl.alloc(tl.float16, [BLOCK_N // 16, BLOCK_M // 16, 16, 16],
                    _address_space=al.ascend_address_space.L1)
    pv_ub = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL],
                     _address_space=al.ascend_address_space.UB)

    for tile_id in tl.range(pid, total_tiles, num_pid):
        task_m_idx = tile_id % num_blocks_m
        task_hz_idx = tile_id // num_blocks_m
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(
            tl.int64) * stride_qh
        if IS_CAUSAL:
            loop_end = (task_m_idx + 1) * BLOCK_M
        else:
            loop_end = N_CTX

        # Block pointers for Q, K, V (Q/K/V tile bases for this tile)
        q_block_ptr = tl.make_block_ptr(
            base=Q + qkv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            base=K + qkv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=V + qkv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )

        # ---------------- Cube side (one scope per tile) ----------------
        with al.scope(core_mode="cube"):
            # q tile stays in SRAM for the whole start_n loop
            q = tl.load(q_block_ptr)

            for start_n in tl.range(0, loop_end, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                k = tl.load(tl.advance(k_block_ptr, (start_n, 0)))
                qk_l0c = tl.dot(q, tl.trans(k))
                al.fixpipe(qk_l0c,
                           qk_ub,
                           dma_mode=al.FixpipeDMAMode.NZ2ND,
                           dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX,
                                  al.PIPE.PIPE_V)

                # Wait for P in L1 (NZ fractal), then PV matmul
                al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3,
                                   al.PIPE.PIPE_MTE1)
                v = tl.load(tl.advance(v_block_ptr, (start_n, 0)))
                pv_l0c = tl.dot(
                    bl.to_tensor(p_l1, target_shape=[BLOCK_M, BLOCK_N]), v)
                al.fixpipe(pv_l0c,
                           pv_ub,
                           dma_mode=al.FixpipeDMAMode.NZ2ND,
                           dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_FIX,
                                  al.PIPE.PIPE_V)

                # No "consumed" signal needed: in-order pipes + data deps imply
                # safety — QK(i+1) follows wait(1)(i) which implies vector has
                # read qk_ub(i); PV(i+1) follows wait(1)(i+1) which implies
                # vector finished iteration i (incl. reading pv_ub) entirely.

        # ---------------- Vector side (one scope per tile) ----------------
        with al.scope(core_mode="vector"):
            sub_id = al.sub_vec_id()
            row0 = sub_id * SUB_M
            offs_dv = tl.arange(0, BLOCK_DMODEL)
            offs_dv = tl.max_contiguous(tl.multiple_of(offs_dv, 16),
                                        BLOCK_DMODEL)

            # per-core state: each vector core owns its SUB_M rows outright
            m_i = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([SUB_M], dtype=tl.float32)
            acc = tl.zeros([SUB_M, BLOCK_DMODEL], dtype=tl.float32)

            for start_n in tl.range(0, loop_end, BLOCK_N):
                al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX,
                                   al.PIPE.PIPE_V)

                qk_sub = bl.to_tensor(
                    qk_ub)  # [SUB_M, BLOCK_N] this core's half
                offs_m_sub = task_m_idx * BLOCK_M + row0 + tl.arange(0, SUB_M)
                qk_sub *= sm_scale
                if IS_CAUSAL:
                    mask_offsets = (offs_m_sub[:, None] * N_CTX + start_n +
                                    tl.arange(0, BLOCK_N)[None, :])
                    causal_mask = tl.load(ATTEN_MASK + mask_offsets)
                    qk_sub += tl.where(causal_mask != 0, -1.0e4, 0.0)

                # online softmax update on this core's rows
                m_ij = tl.max(qk_sub, 1, propagate_nan=True)
                m_new = tl.maximum(m_i,
                                   m_ij,
                                   propagate_nan=tl.PropagateNan.ALL)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk_sub - m_new[:, None])
                l_ij = tl.sum(p, 1)
                l_new = alpha * l_i + l_ij

                # Hand P to Cube in L1 (NZ fractal) via reshape + permute:
                # ND [SUB_M, BLOCK_N] -> (SUB_M, BLOCK_N//16, 16)
                # -> permute (1,0,2) -> [BLOCK_N//16, SUB_M, 16]
                # -> [BLOCK_N//16, SUB_M//16, 16, 16]
                p16 = p.to(tl.float16)
                p_perm = tl.permute(p16.reshape(SUB_M, BLOCK_N // 16, 16),
                                    (1, 0, 2))
                p_nz = p_perm.reshape(BLOCK_N // 16, SUB_M // 16, 16, 16)
                p_nz_buf = bl.to_buffer(p_nz, space=al.ascend_address_space.UB)
                p_l1_sub = bl.subview(p_l1, (0, row0 // 16, 0, 0),
                                      (BLOCK_N // 16, SUB_M // 16, 16, 16),
                                      (1, 1, 1, 1))
                al.copy_from_ub_to_l1(p_nz_buf, p_l1_sub)
                al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3,
                                  al.PIPE.PIPE_MTE1)

                al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_FIX,
                                   al.PIPE.PIPE_V)
                pv = bl.to_tensor(
                    pv_ub)  # [SUB_M, BLOCK_DMODEL] this core's half
                acc = acc * alpha[:, None] + pv
                l_i = l_new
                m_i = m_new

            # Epilogue: store this core's rows via block pointers
            o_block_ptr_sub = tl.make_block_ptr(
                base=Out + qkv_offset + BLOCK_DMODEL * sub_id * SUB_M,
                shape=(N_CTX, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(task_m_idx * BLOCK_M, 0),
                block_shape=(SUB_M, BLOCK_DMODEL),
                order=(1, 0),
            )
            tl.store(o_block_ptr_sub, (acc / l_i[:, None]).to(tl.float16))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=True, BLOCK_M=64, BLOCK_N=64):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        num_blocks_m = triton.cdiv(q.shape[2], BLOCK_M)
        total_tiles = num_blocks_m * q.shape[0] * q.shape[1]
        try:
            device = torch.npu.current_device()
            num_aicore = driver.active.utils.get_device_properties(
                device)["num_aicore"]
        except Exception:
            # Conservative fallback keeps coreDim legal if properties are unavailable.
            num_aicore = 24
        grid = (min(num_aicore, total_tiles), )
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]),
                          device=q.device,
                          dtype=torch.float32)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]),
                        device=q.device,
                        dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]),
                        device=q.device,
                        dtype=torch.float32)
        if causal:
            atten_mask = _causal_mask(q.shape[2], q.device)
        else:
            atten_mask = torch.empty((1, ), device=q.device, dtype=torch.bool)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
            atten_mask,
            sm_scale,
            tmp,
            L,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            SUB_M=BLOCK_M // 2,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=2,
            # Manual sync_block scheme: the default auto block-sync injection
            # deadlocks (or silently corrupts) in-loop v->c syncs.
            disable_auto_inject_block_sync=True,
            vf_merge_level=1,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


attention = _attention.apply
