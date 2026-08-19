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
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def vec_prefree_pv_ub():
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def vec_postwait_p_l1():
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                       al.PIPE.PIPE_MTE3)
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                       al.PIPE.PIPE_MTE3)


@triton.jit
def cube_prefree_p_l1():
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                      al.PIPE.PIPE_MTE3)
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                      al.PIPE.PIPE_MTE3)


@triton.jit
def cube_postwait_s_ub():
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def cube_postwait_pv_ub():
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)


@triton.jit
def _qk_matmul(q, k_block_ptr, start_n, qk_ub0, qk_ub1, sid):
    # Cube side: K chunk load -> QK^T dot -> fixpipe to UB (ROW_SPLIT halves)
    # -> signal vector (event 0).
    kc = tl.load(tl.advance(k_block_ptr, (start_n, 0)))
    qk_c = tl.dot(q, tl.trans(kc))
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    if (sid % 2) == 0:
        qk_ub = bl.to_tensor(qk_ub0)
    else:
        qk_ub = bl.to_tensor(qk_ub1)

    al.fixpipe(qk_c,
               bl.to_buffer(qk_ub, space=al.ascend_address_space.UB),
               dma_mode=al.FixpipeDMAMode.NZ2ND,
               dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _pv_matmul(v_block_ptr, p_l1_0, p_l1_1, pv_ub0, pv_ub1, start_n, pvid,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_DMODEL: tl.constexpr):
    # Cube side: wait for P in L1 (event 4) -> V block load -> PV dot ->
    # release the p_l1 slot (event 6) -> wait for the pv slot (event 10) ->
    # fixpipe to UB -> signal vector (event 8).
    al.sync_block_wait("vector", "cube", 4, al.PIPE.PIPE_MTE3,
                       al.PIPE.PIPE_MTE1)

    if (pvid % 2) == 0:
        p_l1 = bl.to_tensor(p_l1_0, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub0)
    else:
        p_l1 = bl.to_tensor(p_l1_1, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub1)

    vc = tl.load(tl.advance(v_block_ptr, (start_n, 0)))
    pv_c = tl.dot(p_l1, vc)
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                      al.PIPE.PIPE_MTE3)

    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.fixpipe(pv_c,
               bl.to_buffer(pv_ub, space=al.ascend_address_space.UB),
               dma_mode=al.FixpipeDMAMode.NZ2ND,
               dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    al.sync_block_set("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)


@triton.jit
def _softmax_rows_bn64(qk, causal_mask, m_i, sm_scale, SUB_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, IS_CAUSAL: tl.constexpr):
    # Row-wise softmax stats over [SUB_M, BLOCK_N] with single-shot (256B f32)
    # row ops, as an outlined SIMD vector function: loop 1 -> per-row max m_ij;
    # loop 2 -> per-row exp/sum l_ij and the unnormalized P tile.
    qk_scale = bl.alloc(tl.float32, [SUB_M, BLOCK_N],
                        al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)
    p = bl.alloc(tl.float16, [BLOCK_N // 16, SUB_M // 16 * 16, 16],
                 al.ascend_address_space.UB)
    p = bl.to_tensor(p)
    with al.scope(vector_mode="simd", outline=True):
        m_ij = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        l_ij = tl.zeros([SUB_M], dtype=tl.float32)
        for i in range(SUB_M):
            row = al.extract_slice(qk, (i, 0), (1, BLOCK_N), (1, 1))
            row = row * sm_scale
            if IS_CAUSAL:
                mask = al.extract_slice(causal_mask, (i, 0), (1, BLOCK_N),
                                        (1, 1))
                row += tl.where(mask != 0, -1.0e4, 0.0)
            qk_scale = al.insert_slice(qk_scale, row, (i, 0), (1, BLOCK_N),
                                       (1, 1))
            m_row = tl.max(row, 1, propagate_nan=True)  # [1]
            m_ij = al.insert_slice(m_ij, m_row, (i, ), (1, ), (1, ))
        m_ij = tl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        for i in range(SUB_M):
            row = al.extract_slice(qk_scale, (i, 0), (1, BLOCK_N), (1, 1))
            m_r = al.extract_slice(m_ij, (i, ), (1, ), (1, ))  # [1]
            p_row = tl.exp(row - m_r)
            l_row = tl.sum(p_row, 1)  # [1]
            l_ij = al.insert_slice(l_ij, l_row, (i, ), (1, ), (1, ))
            p = al.insert_slice(
                p,
                p_row.reshape(BLOCK_N // 16, 1, 16).to(tl.float16), (0, i, 0),
                (BLOCK_N // 16, 1, 16), (1, 1, 1))
    return m_ij, l_ij, p


@triton.jit
def _softmax_rows_bn128(qk, causal_mask, m_i, sm_scale, SUB_M: tl.constexpr,
                        BLOCK_N: tl.constexpr, IS_CAUSAL: tl.constexpr):
    # BLOCK_N=128: split rows into legal 64-wide fp32 halves. Keep the max
    # loop and exp/sum/P loop as separate outlined SIMD functions.
    HALF: tl.constexpr = BLOCK_N // 2
    m_ij = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij)
    tmp_max = bl.alloc(tl.float32, [SUB_M], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)
    qk_scale = bl.alloc(tl.float32, [SUB_M, BLOCK_N],
                        al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)
    p = bl.alloc(tl.float16, [BLOCK_N // 16, SUB_M // 16 * 16, 16],
                 al.ascend_address_space.UB)
    p = bl.to_tensor(p)
    with al.scope(vector_mode="simd", outline=True):
        m_ij = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        tmp_max = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
        for i in range(SUB_M):
            off_hi = (i * 0) + HALF
            row_lo = al.extract_slice(qk, (i, 0), (1, HALF), (1, 1)) * sm_scale
            row_hi = al.extract_slice(qk, (i, off_hi), (1, HALF),
                                      (1, 1)) * sm_scale
            if IS_CAUSAL:
                mask_lo = al.extract_slice(causal_mask, (i, 0), (1, HALF),
                                           (1, 1))
                mask_hi = al.extract_slice(causal_mask, (i, HALF), (1, HALF),
                                           (1, 1))
                row_lo += tl.where(mask_lo != 0, -1.0e4, 0.0)
                row_hi += tl.where(mask_hi != 0, -1.0e4, 0.0)
            qk_scale = al.insert_slice(qk_scale, row_lo, (i, 0), (1, HALF),
                                       (1, 1))
            qk_scale = al.insert_slice(qk_scale, row_hi, (i, off_hi),
                                       (1, HALF), (1, 1))
            row_max = tl.maximum(row_lo,
                                 row_hi,
                                 propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)
            tmp_max = al.insert_slice(tmp_max, row_max_agg, (i, ), (1, ),
                                      (1, ))
        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)
    with al.scope(vector_mode="simd", outline=True):
        l_ij = tl.zeros([SUB_M], dtype=tl.float32)
        for i in range(SUB_M):
            off_hi = (i * 0) + HALF
            m_r = al.extract_slice(m_ij, (i, ), (1, ), (1, ))
            row_lo = al.extract_slice(qk_scale, (i, 0), (1, HALF), (1, 1))
            row_hi = al.extract_slice(qk_scale, (i, off_hi), (1, HALF), (1, 1))
            p_lo = tl.exp(row_lo - m_r)
            p_hi = tl.exp(row_hi - m_r)
            l_row = tl.sum(p_lo, 1) + tl.sum(p_hi, 1)
            l_ij = al.insert_slice(l_ij, l_row, (i, ), (1, ), (1, ))
            p = al.insert_slice(p,
                                p_lo.reshape(HALF // 16, 1, 16).to(tl.float16),
                                (0, i, 0), (HALF // 16, 1, 16), (1, 1, 1))
            p = al.insert_slice(p,
                                p_hi.reshape(HALF // 16, 1, 16).to(tl.float16),
                                (HALF // 16, i, 0), (HALF // 16, 1, 16),
                                (1, 1, 1))
    return m_ij, l_ij, p


@triton.jit
def _softmax(qk_ub0, qk_ub1, p_l1_0, p_l1_1, m_i, l_i, alpha_scale_a,
             alpha_scale_b, sm_scale, row0, m_base, start_n, sid, ATTEN_MASK,
             N_CTX: tl.constexpr, SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
             IS_CAUSAL: tl.constexpr):
    # Vector side: everything except the acc update — wait for QK (event 0),
    # row-wise stats + P via the outlined SIMD sub-functions, online running
    # update, free the qk slot (event 2), NZ fractal repack, wait for the
    # p_l1 slot (event 6), UB->L1 copy, signal cube (event 4).
    al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    if (sid % 2) == 0:
        qk = bl.to_tensor(qk_ub0)  # [SUB_M, BLOCK_N] this core's half
    else:
        qk = bl.to_tensor(qk_ub1)
    causal_mask = tl.zeros([SUB_M, BLOCK_N], dtype=tl.int8)
    if IS_CAUSAL:
        mask_offsets = ((m_base + tl.arange(0, SUB_M))[:, None] * N_CTX +
                        start_n + tl.arange(0, BLOCK_N)[None, :])
        causal_mask = tl.load(ATTEN_MASK + mask_offsets)
    if BLOCK_N == 128:
        m_ij, l_ij, p = _softmax_rows_bn128(qk, causal_mask, m_i, sm_scale,
                                            SUB_M, BLOCK_N, IS_CAUSAL)
    else:
        m_ij, l_ij, p = _softmax_rows_bn64(qk, causal_mask, m_i, sm_scale,
                                           SUB_M, BLOCK_N, IS_CAUSAL)
    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.exp(m_i - m_ij)
        l_new = alpha * l_i + l_ij
        m_i = m_ij
    if (sid % 2) == 0:
        alpha_scale_a = alpha
    else:
        alpha_scale_b = alpha
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    p_nz = p.reshape(BLOCK_N // 16, SUB_M // 16, 16, 16)
    p_nz_buf = bl.to_buffer(p_nz, space=al.ascend_address_space.UB)
    # wait until the cube's PV dot has released this p_l1 slot
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1,
                       al.PIPE.PIPE_MTE3)
    if (sid % 2) == 0:
        al.copy_from_ub_to_l1(
            p_nz_buf,
            bl.subview(p_l1_0, (0, row0 // 16, 0, 0),
                       (BLOCK_N // 16, SUB_M // 16, 16, 16), (1, 1, 1, 1)))
    else:
        al.copy_from_ub_to_l1(
            p_nz_buf,
            bl.subview(p_l1_1, (0, row0 // 16, 0, 0),
                       (BLOCK_N // 16, SUB_M // 16, 16, 16), (1, 1, 1, 1)))
    al.sync_block_set("vector", "cube", 4, al.PIPE.PIPE_MTE3,
                      al.PIPE.PIPE_MTE1)
    return m_ij, l_new, alpha_scale_a, alpha_scale_b


@triton.jit
def _acc_update(acc, pv_ub0, pv_ub1, alpha_scale0, alpha_scale1, pvid):
    # Vector side: wait for PV (event 8), update the unnormalized numerator,
    # then free the pv slot (event 10).
    al.sync_block_wait("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    if (pvid % 2) == 0:
        pv = bl.to_tensor(pv_ub0)
        acc = acc * alpha_scale0[:, None] + pv
    else:
        pv = bl.to_tensor(pv_ub1)
        acc = acc * alpha_scale1[:, None] + pv

    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    return acc


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

    # Stage 2 ping-pong: slot pairs for every CV handoff buffer (chunk A -> slot 0,
    # chunk B -> slot 1), so Cube runs one chunk ahead of Vector and vice versa.
    # UB buffers are per-vector-core halves (fixpipe ROW_SPLIT drains L0C rows
    # [0, M/2) -> AIV0 UB, [M/2, M) -> AIV1 UB). p_l1_*: P in NZ fractal for PV.
    qk_ub0 = bl.alloc(tl.float32, [SUB_M, BLOCK_N],
                      _address_space=al.ascend_address_space.UB)
    qk_ub1 = bl.alloc(tl.float32, [SUB_M, BLOCK_N],
                      _address_space=al.ascend_address_space.UB)
    p_l1_0 = bl.alloc(tl.float16, [BLOCK_N // 16, BLOCK_M // 16, 16, 16],
                      _address_space=al.ascend_address_space.L1)
    p_l1_1 = bl.alloc(tl.float16, [BLOCK_N // 16, BLOCK_M // 16, 16, 16],
                      _address_space=al.ascend_address_space.L1)
    pv_ub0 = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL],
                      _address_space=al.ascend_address_space.UB)
    pv_ub1 = bl.alloc(tl.float32, [SUB_M, BLOCK_DMODEL],
                      _address_space=al.ascend_address_space.UB)

    # Cube and Vector each own the persistent tile loop. Their tile sequences
    # are identical and communicate only through the existing CV buffers/events.
    with al.scope(core_mode="cube"):
        for tile_id in tl.range(pid, total_tiles, num_pid):
            task_m_idx = tile_id % num_blocks_m
            task_hz_idx = tile_id // num_blocks_m
            off_z = task_hz_idx // H
            off_h = task_hz_idx % H
            qkv_offset = (off_z.to(tl.int64) * stride_qz +
                          off_h.to(tl.int64) * stride_qh)
            if IS_CAUSAL:
                loop_end = (task_m_idx + 1) * BLOCK_M
            else:
                loop_end = N_CTX

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

            q = tl.load(q_block_ptr)
            cube_prefree_p_l1()
            sid = 0
            pvid = 0
            for start_n in tl.range(0, loop_end, 2 * BLOCK_N):
                batch_size = min(2, (loop_end - start_n) // BLOCK_N)
                for batch_idx in range(0, batch_size, 1):
                    _qk_matmul(q, k_block_ptr, start_n + batch_idx * BLOCK_N,
                               qk_ub0, qk_ub1, sid)
                    sid += 1
                for batch_idx in range(0, batch_size, 1):
                    _pv_matmul(v_block_ptr, p_l1_0, p_l1_1, pv_ub0, pv_ub1,
                               start_n + batch_idx * BLOCK_N, pvid, BLOCK_M,
                               BLOCK_N, BLOCK_DMODEL)
                    pvid += 1
            cube_postwait_s_ub()
            cube_postwait_pv_ub()

    with al.scope(core_mode="vector"):
        for tile_id in tl.range(pid, total_tiles, num_pid):
            task_m_idx = tile_id % num_blocks_m
            task_hz_idx = tile_id // num_blocks_m
            off_z = task_hz_idx // H
            off_h = task_hz_idx % H
            qkv_offset = (off_z.to(tl.int64) * stride_qz +
                          off_h.to(tl.int64) * stride_qh)
            if IS_CAUSAL:
                loop_end = (task_m_idx + 1) * BLOCK_M
            else:
                loop_end = N_CTX

            sub_id = al.sub_vec_id()
            row0 = sub_id * SUB_M
            m_i = tl.zeros([SUB_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([SUB_M], dtype=tl.float32)
            acc = tl.zeros([SUB_M, BLOCK_DMODEL], dtype=tl.float32)
            m_base = (task_m_idx * BLOCK_M + row0).to(tl.int32)
            sid = 0
            pvid = 0
            alpha_scale_a = tl.zeros([SUB_M], dtype=tl.float32)
            alpha_scale_b = tl.zeros([SUB_M], dtype=tl.float32)
            al.compile_hint(alpha_scale_a, "alpha_scale_a")
            al.compile_hint(alpha_scale_b, "alpha_scale_b")
            vec_prefree_s_ub()
            vec_prefree_pv_ub()

            for start_n in tl.range(0, loop_end, 2 * BLOCK_N):
                batch_size = min(2, (loop_end - start_n) // BLOCK_N)
                for batch_idx in range(0, batch_size, 1):
                    (m_i, l_i, alpha_scale_a, alpha_scale_b) = _softmax(
                        qk_ub0, qk_ub1, p_l1_0, p_l1_1, m_i, l_i,
                        alpha_scale_a, alpha_scale_b, sm_scale, row0, m_base,
                        start_n + batch_idx * BLOCK_N, sid, ATTEN_MASK, N_CTX,
                        SUB_M, BLOCK_N, IS_CAUSAL)
                    sid += 1
                for batch_idx in range(0, batch_size, 1):
                    acc = _acc_update(acc, pv_ub0, pv_ub1, alpha_scale_a,
                                      alpha_scale_b, pvid)
                    pvid += 1

            o_block_ptr_sub = tl.make_block_ptr(
                base=Out + qkv_offset + BLOCK_DMODEL * sub_id * SUB_M,
                shape=(N_CTX, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(task_m_idx * BLOCK_M, 0),
                block_shape=(SUB_M, BLOCK_DMODEL),
                order=(1, 0),
            )
            tl.store(o_block_ptr_sub, (acc / l_i[:, None]).to(tl.float16))
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
        q_ptrs = Q + (offs_qm[:, None] * stride_qm +
                      offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn +
                      offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm +
                      offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm +
                        offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm +
                        offs_k[None, :] * stride_qk)
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
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk,
                          float("-inf"))
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
        dv_ptrs = DV + (offs_n[:, None] * stride_qm +
                        offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn +
                        offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


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
            # Manual sync_block scheme: the default auto block-sync injection
            # deadlocks (or silently corrupts) in-loop v->c syncs.
            multibuffer=True,
            disable_auto_inject_block_sync=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            enable_auto_vectorize_v2=False,
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
        q, k, v, o, denom, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(denom)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o,
            do,
            denom,
            do_scaled,
            delta,
            BLOCK_M=ctx.BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )

        # NOTE: kernel currently buggy for other values of `num_warps`
        num_warps = 8
        _bwd_kernel[(ctx.grid[1], )](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            denom,
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
