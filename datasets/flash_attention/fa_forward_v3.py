# [2022-10-23] Downloaded from https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py
# for benchmarking.
# We fixed a few dtype cast to make it work for bf16

"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
import triton.runtime.driver as driver


@triton.jit
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def vec_prefree_pv_ub():
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def vec_postwait_p_l1():
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)


@triton.jit
def cube_prefree_p_l1():
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

@triton.jit
def cube_postwait_s_ub():
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def cube_postwait_pv_ub():
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def _qk_matmul(q, k_ptr, start_n, qk_ub0, qk_ub1, qk_l0c, sid):
    # Cube side: K chunk load -> QK^T dot -> fixpipe to UB (ROW_SPLIT halves)
    # -> signal vector (event 0).
    kc = tl.load(k_ptr)
    qk_c = tl.dot(q, tl.trans(kc))
    bl.to_buffer(qk_c, bind_buffer=qk_l0c)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    if (sid % 2) == 0:
        qk_ub = bl.to_tensor(qk_ub0)
    else:
        qk_ub = bl.to_tensor(qk_ub1)

    al.fixpipe(qk_c, bl.to_buffer(qk_ub, space=al.ascend_address_space.UB), dma_mode=al.FixpipeDMAMode.NZ2ND,
               dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _pv_matmul(v_ptr, p_l1_0, p_l1_1, pv_ub0, pv_ub1, pv_l0c, pvid,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_DMODEL: tl.constexpr):
    # Cube side: wait for P in L1 (event 4) -> raw V chunk load (in-loop
    # block-ptr V loads hit the PlanMemory empty-addrs bug on 9.0-gen
    # bishengir; stride_vk = N(row) stride, stride_vn = D(col) stride) ->
    # PV dot -> release the p_l1 slot (event 6) -> wait for the pv slot
    # (event 10) -> fixpipe to UB -> signal vector (event 8).
    al.sync_block_wait("vector", "cube", 4, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

    if (pvid % 2) == 0:
        p_l1 = bl.to_tensor(p_l1_0, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub1)
    else:
        p_l1 = bl.to_tensor(p_l1_1, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub0)

    vc = tl.load(v_ptr)
    pv_c = tl.dot(p_l1, vc)
    bl.to_buffer(pv_c, bind_buffer=pv_l0c)
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.fixpipe(pv_c, bl.to_buffer(pv_ub, space=al.ascend_address_space.UB), dma_mode=al.FixpipeDMAMode.NZ2ND,
               dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    al.sync_block_set("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _softmax_rows_bn64(qk, p_buf, m_i, sm_scale, m_base, start_n,
                       SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       IS_CAUSAL: tl.constexpr, NEED_UPDATE: tl.constexpr):
    if NEED_UPDATE:
        tmp_max = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        tmp_max = bl.to_tensor(tmp_max)
    qk_scale = bl.alloc(tl.float32, [SUB_M, BLOCK_N], al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)
    p = bl.to_tensor(p_buf)
    with al.scope(vector_mode="simd", outline=True):
        m_ij = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        if NEED_UPDATE:
            tmp_max = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        l_ij = tl.zeros([SUB_M], dtype=tl.float32)
        for i in range(SUB_M):
            row = al.extract_slice(qk, (i, 0), (1, BLOCK_N), (1, 1))
            row = row * sm_scale
            if IS_CAUSAL:
                cols = start_n + tl.arange(0, BLOCK_N)
                row += tl.where((m_base + i) >= cols[None, :], 0.0, float("-inf"))
            qk_scale = al.insert_slice(qk_scale, row, (i, 0), (1, BLOCK_N), (1, 1))
            m_row = tl.max(row, 1)
            if NEED_UPDATE:
                tmp_max = al.insert_slice(tmp_max, m_row, (i,), (1,), (1,))
            else:
                m_ij = al.insert_slice(m_ij, m_row, (i,), (1,), (1,))
        if NEED_UPDATE:
            m_ij = tl.maximum(m_i, tmp_max)
        for i in range(SUB_M):
            row = al.extract_slice(qk_scale, (i, 0), (1, BLOCK_N), (1, 1))
            m_r = al.extract_slice(m_ij, (i,), (1,), (1,))
            p_row = tl.exp(row - m_r)
            l_row = tl.sum(p_row, 1)
            l_ij = al.insert_slice(l_ij, l_row, (i,), (1,), (1,))
            p = al.insert_slice(
                p,
                p_row.reshape(BLOCK_N // 16, 1, 16).to(tl.float16),
                (0, i, 0),
                (BLOCK_N // 16, 1, 16),
                (1, 1, 1))
    al.copy(bl.to_buffer(p, space=al.ascend_address_space.UB), p_buf)
    return m_ij, l_ij


@triton.jit
def _softmax_rows_bn128(qk, p_buf, m_i, sm_scale, m_base, start_n,
                        SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        IS_CAUSAL: tl.constexpr, NEED_UPDATE: tl.constexpr):
    HALF: tl.constexpr = BLOCK_N // 2
    m_ij = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij)
    if NEED_UPDATE:
        tmp_max = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        tmp_max = bl.to_tensor(tmp_max)
    qk_scale = bl.alloc(tl.float32, [SUB_M, BLOCK_N], al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)
    p = bl.to_tensor(p_buf)
    with al.scope(vector_mode="simd", outline=True):
        m_ij = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        if NEED_UPDATE:
            tmp_max = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        for i in range(SUB_M):
            off_hi = (i * 0) + HALF
            row_lo = al.extract_slice(qk, (i, 0), (1, HALF), (1, 1)) * sm_scale
            row_hi = al.extract_slice(qk, (i, off_hi), (1, HALF), (1, 1)) * sm_scale
            if IS_CAUSAL:
                cols = start_n + tl.arange(0, HALF)
                row_lo += tl.where((m_base + i) >= cols[None, :], 0.0, float("-inf"))
                row_hi += tl.where((m_base + i) >= (cols + HALF)[None, :], 0.0, float("-inf"))
            qk_scale = al.insert_slice(qk_scale, row_lo, (i, 0), (1, HALF), (1, 1))
            qk_scale = al.insert_slice(qk_scale, row_hi, (i, off_hi), (1, HALF), (1, 1))
            row_max = tl.maximum(row_lo, row_hi)
            row_max_agg = tl.max(row_max, 1)
            if NEED_UPDATE:
                tmp_max = al.insert_slice(tmp_max, row_max_agg, (i,), (1,), (1,))
            else:
                m_ij = al.insert_slice(m_ij, row_max_agg, (i,), (1,), (1,))
        if NEED_UPDATE:
            m_ij = tl.maximum(m_i, tmp_max)
    with al.scope(vector_mode="simd", outline=True):
        l_ij = tl.zeros([SUB_M], dtype=tl.float32)
        for i in range(SUB_M):
            off_hi = (i * 0) + HALF
            m_r = al.extract_slice(m_ij, (i,), (1,), (1,))
            row_lo = al.extract_slice(qk_scale, (i, 0), (1, HALF), (1, 1))
            row_hi = al.extract_slice(qk_scale, (i, off_hi), (1, HALF), (1, 1))
            p_lo = tl.exp(row_lo - m_r)
            p_hi = tl.exp(row_hi - m_r)
            l_row = tl.sum(p_lo, 1) + tl.sum(p_hi, 1)
            l_ij = al.insert_slice(l_ij, l_row, (i,), (1,), (1,))
            p = al.insert_slice(p, p_lo.reshape(HALF // 16, 1, 16).to(tl.float16),
                                (0, i, 0), (HALF // 16, 1, 16), (1, 1, 1))
            p = al.insert_slice(p, p_hi.reshape(HALF // 16, 1, 16).to(tl.float16),
                                (HALF // 16, i, 0), (HALF // 16, 1, 16), (1, 1, 1))
    al.copy(bl.to_buffer(p, space=al.ascend_address_space.UB), p_buf)
    return m_ij, l_ij


@triton.jit
def _online_update(m_i, l_i, m_new, l_ij, SUB_M: tl.constexpr):
    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.exp(m_i - m_new)
        l_new = alpha * l_i + l_ij
        m_i = m_new
    return m_new, l_new, alpha


@triton.jit
def _softmax_with_mask_with_update(qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, m_base, start_n,
                                   SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m_i = bl.to_tensor(m_i_buf)
    l_i = bl.to_tensor(l_i_buf)
    if BLOCK_N == 64:
        m_new, l_ij = _softmax_rows_bn64(
            qk, p_buf, m_i, sm_scale, m_base, start_n,
            SUB_M, BLOCK_N, True, True)
    else:
        m_new, l_ij = _softmax_rows_bn128(
            qk, p_buf, m_i, sm_scale, m_base, start_n,
            SUB_M, BLOCK_N, True, True)
    m_new, l_new, acc_scale = _online_update(
        m_i, l_i, m_new, l_ij, SUB_M)
    al.copy(bl.to_buffer(m_new, space=al.ascend_address_space.UB), m_i_buf)
    al.copy(bl.to_buffer(l_new, space=al.ascend_address_space.UB), l_i_buf)
    al.copy(bl.to_buffer(acc_scale, space=al.ascend_address_space.UB), acc_scale_buf)


@triton.jit
def _softmax_no_mask_with_update(qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale,
                                 SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m_i = bl.to_tensor(m_i_buf)
    l_i = bl.to_tensor(l_i_buf)
    if BLOCK_N == 64:
        m_new, l_ij = _softmax_rows_bn64(
            qk, p_buf, m_i, sm_scale, 0, 0,
            SUB_M, BLOCK_N, False, True)
    else:
        m_new, l_ij = _softmax_rows_bn128(
            qk, p_buf, m_i, sm_scale, 0, 0,
            SUB_M, BLOCK_N, False, True)
    m_new, l_new, acc_scale = _online_update(
        m_i, l_i, m_new, l_ij, SUB_M)
    al.copy(bl.to_buffer(m_new, space=al.ascend_address_space.UB), m_i_buf)
    al.copy(bl.to_buffer(l_new, space=al.ascend_address_space.UB), l_i_buf)
    al.copy(bl.to_buffer(acc_scale, space=al.ascend_address_space.UB), acc_scale_buf)


@triton.jit
def _softmax_with_mask_no_update(qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, m_base, start_n,
                                 SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m_init = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
    if BLOCK_N == 64:
        m_new, l_new = _softmax_rows_bn64(
            qk, p_buf, m_init, sm_scale, m_base, start_n,
            SUB_M, BLOCK_N, True, False)
    else:
        m_new, l_new = _softmax_rows_bn128(
            qk, p_buf, m_init, sm_scale, m_base, start_n,
            SUB_M, BLOCK_N, True, False)
    al.copy(bl.to_buffer(m_new, space=al.ascend_address_space.UB), m_i_buf)
    al.copy(bl.to_buffer(l_new, space=al.ascend_address_space.UB), l_i_buf)
    acc_scale = tl.zeros([SUB_M], dtype=tl.float32)
    al.copy(bl.to_buffer(acc_scale, space=al.ascend_address_space.UB), acc_scale_buf)


@triton.jit
def _softmax_no_mask_no_update(qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale,
                               SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m_init = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
    if BLOCK_N == 64:
        m_new, l_new = _softmax_rows_bn64(
            qk, p_buf, m_init, sm_scale, 0, 0,
            SUB_M, BLOCK_N, False, False)
    else:
        m_new, l_new = _softmax_rows_bn128(
            qk, p_buf, m_init, sm_scale, 0, 0,
            SUB_M, BLOCK_N, False, False)
    al.copy(bl.to_buffer(m_new, space=al.ascend_address_space.UB), m_i_buf)
    al.copy(bl.to_buffer(l_new, space=al.ascend_address_space.UB), l_i_buf)
    acc_scale = tl.zeros([SUB_M], dtype=tl.float32)
    al.copy(bl.to_buffer(acc_scale, space=al.ascend_address_space.UB), acc_scale_buf)


@triton.jit
def softmax_vf_select(need_mask, need_update, qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale,
                      m_base, start_n,
                      SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    if need_mask & need_update:
        _softmax_with_mask_with_update(
            qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, m_base, start_n, SUB_M, BLOCK_N)
    elif need_mask & ~need_update:
        _softmax_with_mask_no_update(
            qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, m_base, start_n, SUB_M, BLOCK_N)
    elif ~need_mask & need_update:
        _softmax_no_mask_with_update(
            qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, SUB_M, BLOCK_N)
    else:
        _softmax_no_mask_no_update(
            qk, p_buf, m_i_buf, l_i_buf, acc_scale_buf, sm_scale, SUB_M, BLOCK_N)

@triton.jit
def _softmax_v1(qk_ub0, qk_ub1, p_l1_0, p_l1_1, p,
                m_i_tb0, m_i_tb1, m_i_tb2,
                l_i_tb0, l_i_tb1, l_i_tb2,
                acc_scale0, acc_scale1, acc_scale2,
                sm_scale, vtaskId, v_s1_task_mod3,
                m_base, start_n, cast_dtype,
                need_mask, need_update,
                SUB_M: tl.constexpr, BLOCK_N: tl.constexpr):
    al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    sub_vec_id = al.sub_vec_id()
    if (vtaskId % 2) == 1:
        qk = bl.to_tensor(qk_ub0)
    else:
        qk = bl.to_tensor(qk_ub1)

    scale_slot = (vtaskId - 1) % 3
    if v_s1_task_mod3 == 0 and scale_slot == 0:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb0, l_i_tb0, acc_scale0, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 0 and scale_slot == 1:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb0, l_i_tb0, acc_scale1, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 0 and scale_slot == 2:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb0, l_i_tb0, acc_scale2, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 1 and scale_slot == 0:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb1, l_i_tb1, acc_scale0, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 1 and scale_slot == 1:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb1, l_i_tb1, acc_scale1, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 1 and scale_slot == 2:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb1, l_i_tb1, acc_scale2, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 2 and scale_slot == 0:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb2, l_i_tb2, acc_scale0, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    elif v_s1_task_mod3 == 2 and scale_slot == 1:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb2, l_i_tb2, acc_scale1, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)
    else:
        softmax_vf_select(
            need_mask, need_update, qk, p, m_i_tb2, l_i_tb2, acc_scale2, sm_scale,
            m_base, start_n, SUB_M, BLOCK_N)

    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    p_nz = bl.to_tensor(p).reshape(BLOCK_N // 16, SUB_M // 16, 16, 16)
    p_nz_buf = bl.to_buffer(p_nz, space=al.ascend_address_space.UB)
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    p_task_id = vtaskId - 1
    if (p_task_id % 2) == 0:
        al.copy_from_ub_to_l1(
            p_nz_buf,
            bl.subview(p_l1_0, (0, sub_vec_id * SUB_M // 16, 0, 0),
                       (BLOCK_N // 16, SUB_M // 16, 16, 16), (1, 1, 1, 1)))
    else:
        al.copy_from_ub_to_l1(
            p_nz_buf,
            bl.subview(p_l1_1, (0, sub_vec_id * SUB_M // 16, 0, 0),
                       (BLOCK_N // 16, SUB_M // 16, 16, 16), (1, 1, 1, 1)))
    al.sync_block_set("vector", "cube", 4, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

@triton.jit
def _acc_update(pv_ub0, pv_ub1,
                acc_scale0, acc_scale1, acc_scale2,
                acc_buffer, vtaskId, update_acc,
                SUB_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Vector side: wait for PV (event 8), rescale acc with this chunk's scale,
    # accumulate, then free the pv slot (event 10).
    al.sync_block_wait("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    if (vtaskId % 2) == 0:
        pv = bl.to_tensor(pv_ub0)
    else:
        pv = bl.to_tensor(pv_ub1)

    scale_slot = (vtaskId - 3) % 3
    if scale_slot == 0:
        acc_scale = bl.to_tensor(acc_scale0)
    elif scale_slot == 1:
        acc_scale = bl.to_tensor(acc_scale1)
    else:
        acc_scale = bl.to_tensor(acc_scale2)

    if update_acc:
        acc = bl.to_tensor(acc_buffer)
        acc = acc * acc_scale[:, None] + pv
        bl.to_buffer(acc, bind_buffer=acc_buffer)
    else:
        if (vtaskId % 2) == 0:
            al.copy(pv_ub0, acc_buffer)
        else:
            al.copy(pv_ub1, acc_buffer)

    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def update_s2_loop(taskId_mod3, cur_s2_idx, s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4):
    if taskId_mod3 == 0:
        s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4 = cur_s2_idx, s2_idx_2, sd_idx_3, sd_idx_4
    elif taskId_mod3 == 1:
        s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4 = s2_idx_1, cur_s2_idx, sd_idx_3, sd_idx_4
    elif taskId_mod3 == 2:
        s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4 = s2_idx_1, s2_idx_2, cur_s2_idx, sd_idx_4
    else:    
        s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4 = s2_idx_1, s2_idx_2, sd_idx_3, cur_s2_idx
    return s2_idx_1, s2_idx_2, sd_idx_3, sd_idx_4


@triton.jit
def is_need_update(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_need = not s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_need = not s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_need = not s2_idx_3 == 0
    else:
        is_need = not s2_idx_4 == 0
    return is_need


@triton.jit
def is_last_skv(taskId_mod3, s2_idx_1, s2_size_1, s2_idx_2, s2_size_2, s2_idx_3, s2_size_3, s2_idx_4, s2_size_4):
    if taskId_mod3 == 0:
        is_reach = s2_idx_1 == s2_size_1 - 1
    elif taskId_mod3 == 1:
        is_reach = s2_idx_2 == s2_size_2 - 1
    elif taskId_mod3 == 2:
        is_reach = s2_idx_3 == s2_size_3 - 1
    else:
        is_reach = s2_idx_4 == s2_size_4 - 1

    return is_reach


@triton.jit
def is_first_skv_loop(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_first = s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_first = s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_first = s2_idx_3 == 0
    else:
        is_first = s2_idx_4 == 0

    return is_first


@triton.jit
def get_cur_task(taskId_mod3, b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                 b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                 b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                 b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4):

    if taskId_mod3 == 0: 
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1
    elif taskId_mod3 == 1: 
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2
    elif taskId_mod3 == 2: 
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3
    else:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4

    return cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size

@triton.jit
def update_pos(s1_cur_idx, s1_step, batch_size, head_num, NUM_BLOCKS_M,
               IS_CAUSAL, N_CTX, BLOCK_M, BLOCK_N):
    bn_idx = s1_cur_idx // NUM_BLOCKS_M
    task_m_idx = s1_cur_idx % NUM_BLOCKS_M
    b_idx = bn_idx // head_num
    n_idx = bn_idx % head_num
    if IS_CAUSAL:
        loop_end = (task_m_idx + 1) * BLOCK_M
    else:
        loop_end = N_CTX

    return b_idx, n_idx, task_m_idx, loop_end // BLOCK_N

@triton.jit
def update_task(taskId, task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    s1_task_mod3 = task_cnt % 3
    if (taskId & 3) == 0:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = s1_task_mod3, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 
    elif (taskId & 3) == 1:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, s1_task_mod3, v_s1_task_mod3_3, v_s1_task_mod3_4 
    elif (taskId & 3) == 2:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, v_s1_task_mod3_2, s1_task_mod3, v_s1_task_mod3_4 
    else:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, s1_task_mod3 
    return v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4

@triton.jit
def get_s_task(taskId, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    if (taskId & 3) == 0:
        cur_s1_task_mod3 = v_s1_task_mod3_1
    elif (taskId & 3) == 1:
        cur_s1_task_mod3 = v_s1_task_mod3_2
    elif (taskId & 3) == 2:
        cur_s1_task_mod3 = v_s1_task_mod3_3
    else:
        cur_s1_task_mod3 = v_s1_task_mod3_4

    return cur_s1_task_mod3

@triton.jit
def create_task(taskId, b_idx, n_idx, s1_idx, s2_idx, s2_size,
                b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4):
    if (taskId & 3) == 0:
        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    elif (taskId & 3) == 1:
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = b_idx, n_idx, s1_idx, s2_idx, s2_size       
    elif (taskId & 3) == 2:
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    else:
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = b_idx, n_idx, s1_idx, s2_idx, s2_size

    return (b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
            b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
            b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
            b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

@triton.jit
def create_and_get_basic_pos(s1_cur_idx, s1_step, batch, head_num, s1_start,
                             NUM_BLOCKS_M, IS_CAUSAL, N_CTX, BLOCK_M, BLOCK_N):
    b_idx, n_idx, s1_idx, s2_size = update_pos(
        s1_cur_idx, s1_step, batch, head_num, NUM_BLOCKS_M,
        IS_CAUSAL, N_CTX, BLOCK_M, BLOCK_N)

    return b_idx, n_idx, s1_idx, s2_size

@triton.jit
def get_s_offset(head_num, seqlen, stride, b_idx, n_idx, s_idx):
    s_offset = (b_idx.to(tl.int64) * head_num + n_idx.to(tl.int64)) * seqlen + s_idx.to(tl.int64) * stride

    return s_offset.to(tl.int64)

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
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
    CORE_NUM: tl.constexpr,
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
    num_blocks_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_blocks_m * Z * H

    # Stage 3 preload pipeline: slot pairs for every CV handoff buffer (chunk A -> slot 0,
    # chunk B -> slot 1), so Cube runs one chunk ahead of Vector and vice versa.
    # UB buffers are per-vector-core halves (fixpipe ROW_SPLIT drains L0C rows
    # [0, M/2) -> AIV0 UB, [M/2, M) -> AIV1 UB). p_l1_*: P in NZ fractal for PV.
    qk_ub0 = bl.alloc(tl.float32, [SUB_M, BLOCK_N], _address_space=al.ascend_address_space.UB)
    qk_ub1 = bl.alloc(tl.float32, [SUB_M, BLOCK_N], _address_space=al.ascend_address_space.UB)
    p_l1_0 = bl.alloc(tl.float16, [BLOCK_N // 16, BLOCK_M // 16, 16, 16], _address_space=al.ascend_address_space.L1)
    p_l1_1 = bl.alloc(tl.float16, [BLOCK_N // 16, BLOCK_M // 16, 16, 16], _address_space=al.ascend_address_space.L1)
    pv_ub0 = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL], _address_space=al.ascend_address_space.UB)
    pv_ub1 = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL], _address_space=al.ascend_address_space.UB)

    last_third_loop = 0
    last_second_loop = 0
    last_loop = 0
    multi_core_limit = total_tiles
    preload = 3
    if not IS_CAUSAL:
        last_third_loop = (total_tiles - pid + CORE_NUM - 1) // CORE_NUM * CORE_NUM + pid
        last_second_loop = last_third_loop + CORE_NUM
        last_loop = last_second_loop + CORE_NUM
        multi_core_limit += 3 * CORE_NUM
        start_block, end_block, step = pid, multi_core_limit, CORE_NUM
    else:
        last_third_loop = (total_tiles - pid + CORE_NUM - 1) // CORE_NUM * CORE_NUM + pid
        last_second_loop = last_third_loop + CORE_NUM
        last_loop = last_second_loop + CORE_NUM
        multi_core_limit += 3 * CORE_NUM
        start_block, end_block, step = pid, multi_core_limit, CORE_NUM

    # ---------------- Cube side (one scope per tile) ----------------
    with al.scope(core_mode="cube"):
        # cube keeps own taskInfo[4]
        # c1 always use task_info[cur_idx] as taks producer
        # c2 always use task_info[cur_idx - 2] as taks consumer
        # The deferred consumption masks synchronization issues
        # task_id marks the task_idx as (task_id - consumer_stage)
        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = 0, 0, 0, 0, 0 # task_1 V1
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = 0, 0, 0, 0, 0 # task_2 C1
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = 0, 0, 0, 0, 0
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = 0, 0, 0, 0, 0
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size = 0, 0, 0, 0
        taskId = 0

        cube_prefree_p_l1()

        # Stage 3 manual preload/task-skew schedule. QK is one block ahead
        # of PV on the Cube core: preload QK(0), then issue QK(i) before
        # consuming P(i-1), and finally drain PV(last). Together with the
        # Vector schedule below, steady state overlaps Cube and Vector work
        # while limiting live-stage pressure to the two ping-pong slots.
        for sq_loop_idx in al.parallel(start_block, end_block, step):
            is_last_loop = sq_loop_idx == last_loop
            is_last_second_loop = sq_loop_idx == last_second_loop
            is_last_third_loop = sq_loop_idx == last_third_loop
            not_last = not is_last_loop
            not_last_two = not is_last_second_loop and not is_last_loop
            not_last_three = (not is_last_third_loop) and not_last_two

            if not_last_three:
                cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size = create_and_get_basic_pos(
                    sq_loop_idx, CORE_NUM, Z, H, pid, num_blocks_m,
                    IS_CAUSAL, N_CTX, BLOCK_M, BLOCK_N)

            if not not_last_three:
                cur_s2_size = 1

            s1_task_mod3 = ((sq_loop_idx - pid) // CORE_NUM) % 3
            qk_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.L0C, is_mem_unique=True)
            pv_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_DMODEL], al.ascend_address_space.L0C, is_mem_unique=True)
            q_l1_keep = bl.alloc(tl.float16, [BLOCK_M, BLOCK_DMODEL], al.ascend_address_space.L1)
            for skv_loop_idx in range(0, cur_s2_size):
                # create and push to producer stack
                if not_last_three:
                    (b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                    b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2, 
                    b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3, 
                    b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4) = \
                        create_task(taskId, cur_b_idx, cur_n_idx, cur_s1_idx, skv_loop_idx, cur_s2_size,
                                    b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                                    b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                                    b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                                    b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

                q_rs = get_s_offset(H, N_CTX, BLOCK_M, cur_b_idx, cur_n_idx, cur_s1_idx) + tl.arange(0, BLOCK_M)[:, None]
                q_cs = tl.arange(0, BLOCK_DMODEL)[None, :]
                q_ptr = Q + q_rs * stride_qm + q_cs * stride_qk
                k_rs = get_s_offset(H, N_CTX, BLOCK_N, cur_b_idx, cur_n_idx, skv_loop_idx) + tl.arange(0, BLOCK_N)[:, None]
                k_cs = tl.arange(0, BLOCK_DMODEL)[None, :]
                k_ptr = K + k_rs * stride_kn + k_cs * stride_kk
                if not_last_three:
                    if skv_loop_idx == 0:
                        q_loaded = tl.load(q_ptr)
                        bl.to_buffer(tensor=q_loaded, bind_buffer=q_l1_keep)
                    q = bl.to_tensor(q_l1_keep)
                    _qk_matmul(q, k_ptr, 0, qk_ub0, qk_ub1, qk_l0c, taskId)

                # get c2 task 
                c2_use_b_idx, c2_use_n_idx, c2_use_s1_idx, c2_use_s2_idx, c2_use_s2_size = get_cur_task((taskId + 2) & 3, b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                                                                                                        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                                                                                                        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                                                                                                        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4,
                                                                                                            )

                v_rs = get_s_offset(H, N_CTX, BLOCK_N, c2_use_b_idx, c2_use_n_idx, c2_use_s2_idx) + tl.arange(0, BLOCK_N)[:, None]
                v_cs = tl.arange(0, BLOCK_DMODEL)[None, :]
                v_ptr = V + v_rs * stride_vk + v_cs * stride_vn
                if taskId > 1 and not_last:
                    _pv_matmul(v_ptr, p_l1_0, p_l1_1, pv_ub0, pv_ub1, pv_l0c,
                            taskId, BLOCK_M, BLOCK_N, BLOCK_DMODEL)
                taskId += 1

        # drain the leftover slot-free tokens so the next tile starts at 0
        cube_postwait_s_ub()
        cube_postwait_pv_ub()

    # ---------------- Vector side (one scope per tile) ----------------
    with al.scope(core_mode="vector"):
        # cube keeps own taskInfo[4]
        # v1 always use task_info[cur_idx - 1] as taks producer
        # v2 always use task_info[cur_idx - 3] as taks consumer
        # The deferred consumption masks synchronization issues
        # task_id marks the task_idx as (task_id - consumer_stage)
        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1 = 0, 0, 0, 0, 0 # task_1 V1
        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2 = 0, 0, 0, 0, 0 # task_1 C1
        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3 = 0, 0, 0, 0, 0 
        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4 = 0, 0, 0, 0, 0 
        v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size = 0, 0, 0, 0 

        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = 0, 0, 0, 0
        s1_task_cnt = 0
        vtaskId = 0

        # per-core state: each vector core owns its SUB_M rows outright
        m_i_tb0 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        m_i_tb1 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        m_i_tb2 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)

        l_i_tb0 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        l_i_tb1 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
        l_i_tb2 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)

        acc_scale0 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB, is_mem_unique=True)
        acc_scale1 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB, is_mem_unique=True)
        acc_scale2 = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB, is_mem_unique=True)

        p = bl.alloc(
            tl.float16,
            [BLOCK_N // 16, SUB_M // 16 * 16, 16],
            al.ascend_address_space.UB)
        al.multibuffer(p, 2)

        acc_buffer = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL], al.ascend_address_space.UB, is_mem_unique=True)

        # pre-arm the qk_ub / pv_ub slot-free tokens (both slots start empty)
        vec_prefree_s_ub()
        vec_prefree_pv_ub()
        for sq_loop_idx in al.parallel(start_block, end_block, step):
            v_is_last_loop = sq_loop_idx == last_loop
            v_is_last_second_loop = sq_loop_idx == last_second_loop
            v_is_last_third_loop = sq_loop_idx == last_third_loop
            v_not_last = not v_is_last_loop
            v_not_last_two = not v_is_last_second_loop and not v_is_last_loop
            v_not_last_three = (not v_is_last_third_loop) and v_not_last_two

            if v_not_last_three:
                v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size = create_and_get_basic_pos(
                    sq_loop_idx, CORE_NUM, Z, H, pid, num_blocks_m,
                    IS_CAUSAL, N_CTX, BLOCK_M, BLOCK_N)
            if not v_not_last_three:
                v_cur_s2_size = 1
            s1_task_cnt += 1

            for skv_loop_idx in range(0, v_cur_s2_size):
                if v_not_last_three:
                    (v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                     v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                     v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                     v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4) = \
                        create_task(vtaskId, v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, skv_loop_idx, v_cur_s2_size,
                                    v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                    v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                    v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                    v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                    v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = \
                        update_task(vtaskId, s1_task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)

                # get v1 task 
                v1_use_b_idx, v1_use_n_idx, v1_use_s1_idx, v1_use_s2_idx, v1_use_s2_size = get_cur_task((vtaskId - 1) & 3, 
                                                                                                        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                                                                                        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                                                                                        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                                                                                        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                need_do_v1 = vtaskId > 0 and v_not_last_two
                if need_do_v1:
                    v1_last_skv = v1_use_s2_idx == v1_use_s2_size - 1
                    v1_need_update = v1_use_s2_idx != 0
                    v1_s1_task_mod3 = get_s_task(vtaskId - 1, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)
                    sub_vec_id = al.sub_vec_id()
                    m_base = (v1_use_s1_idx * BLOCK_M + sub_vec_id * SUB_M).to(tl.int32)
                    start_n = v1_use_s2_idx * BLOCK_N
                    need_mask = IS_CAUSAL & ((start_n + BLOCK_N) > m_base)
                    _softmax_v1(
                        qk_ub0, qk_ub1, p_l1_0, p_l1_1, p,
                        m_i_tb0, m_i_tb1, m_i_tb2,
                        l_i_tb0, l_i_tb1, l_i_tb2,
                        acc_scale0, acc_scale1, acc_scale2,
                        sm_scale, vtaskId, v1_s1_task_mod3,
                        m_base, start_n, tl.float16,
                        need_mask, v1_need_update, SUB_M, BLOCK_N)

                # get v2 task
                v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx, v2_use_s2_idx, v2_use_s2_size = get_cur_task((vtaskId + 1) & 3, 
                                                                                        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                                                                        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                                                                        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                                                                        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                
                if vtaskId > 2:
                    update_acc = v2_use_s2_idx != 0
                    _acc_update(pv_ub0, pv_ub1,
                                acc_scale0, acc_scale1, acc_scale2,
                                acc_buffer, vtaskId, update_acc,
                                SUB_M, BLOCK_DMODEL)
       

                v2_last_skv_v2 = v2_use_s2_idx == v2_use_s2_size - 1
                if v2_last_skv_v2:
                    v2_s1_task_mod3 = get_s_task(vtaskId - 3, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)
                    if v2_s1_task_mod3 == 0:
                        l_i = bl.to_tensor(l_i_tb0)
                        m_i = bl.to_tensor(m_i_tb0)
                    elif v2_s1_task_mod3 == 1:
                        l_i = bl.to_tensor(l_i_tb1)
                        m_i = bl.to_tensor(m_i_tb1)
                    else:
                        l_i = bl.to_tensor(l_i_tb2)
                        m_i = bl.to_tensor(m_i_tb2)
                    m_i += tl.math.log(l_i)
                    acc = bl.to_tensor(acc_buffer)
                    acc = acc / l_i[:, None]

                    sub_vec_id = al.sub_vec_id()
                    out_offset = get_s_offset(H, N_CTX, BLOCK_M, v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx) + sub_vec_id * SUB_M
                    o_rs = out_offset + tl.arange(0, SUB_M)[:, None]
                    o_cs = tl.arange(0, BLOCK_DMODEL)[None, :]
                    o_ptrs = Out + o_rs * stride_om + o_cs * stride_on

                    tl.store(o_ptrs, acc.to(tl.float16))

                vtaskId += 1

        vec_postwait_p_l1()


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    L,
    NewDO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
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
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, k, trans_b=True)
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(p.to(do.dtype), do, trans_a=True)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v, trans_b=True)
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds.to(q.dtype), q, trans_a=True)
            # # compute dq
            dq = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq += tl.dot(ds.to(k.dtype), k)
            tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            # # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=True, BLOCK_M=64, BLOCK_N=64):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        # The single accumulator buffer keeps 128x128 within UB for D64 and
        # D128. Callers may still choose smaller tiles for regime-specific
        # performance experiments.

        o = torch.empty_like(q)
        num_blocks_m = triton.cdiv(q.shape[2], BLOCK_M)
        total_tiles = num_blocks_m * q.shape[0] * q.shape[1]
        try:
            device = torch.npu.current_device()
            num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
        except Exception:
            # Conservative fallback keeps coreDim legal if properties are unavailable.
            num_aicore = 24
        grid = (min(num_aicore, total_tiles),)
        tmp = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
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
            CORE_NUM=num_aicore,
            num_warps=num_warps,
            num_stages=2,
            # Manual sync_block scheme: the default auto block-sync injection
            # deadlocks (or silently corrupts) in-loop v->c syncs.
            multibuffer=True,
            sync_solver=True,
            disable_auto_inject_block_sync=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1],)](
            o,
            do,
            l,
            do_scaled,
            delta,
            BLOCK_M=ctx.BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )

        # NOTE: kernel currently buggy for other values of `num_warps`
        num_warps = 8
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            l,
            m,
            delta,
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
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=ctx.BLOCK,
            BLOCK_N=ctx.BLOCK_N,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=1,
        )
        return dq.to(q.dtype), dk, dv, None, None, None, None


attention = _attention.apply