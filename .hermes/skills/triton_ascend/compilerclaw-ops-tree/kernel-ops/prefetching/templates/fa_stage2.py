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
def _pv_matmul(v_base, p_l1_0, p_l1_1, pv_ub0, pv_ub1, start_n, pvid,
               stride_vk, stride_vn, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Cube side: wait for P in L1 (event 4) -> raw V chunk load (in-loop
    # block-ptr V loads hit the PlanMemory empty-addrs bug on 9.0-gen
    # bishengir; stride_vk = N(row) stride, stride_vn = D(col) stride) ->
    # PV dot -> release the p_l1 slot (event 6) -> wait for the pv slot
    # (event 10) -> fixpipe to UB -> signal vector (event 8).
    al.sync_block_wait("vector", "cube", 4, al.PIPE.PIPE_MTE3,
                       al.PIPE.PIPE_MTE1)

    if (pvid % 2) == 0:
        p_l1 = bl.to_tensor(p_l1_0, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub0)
    else:
        p_l1 = bl.to_tensor(p_l1_1, target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub1)

    offs_nc = tl.arange(0, BLOCK_N)
    offs_dc = tl.arange(0, BLOCK_DMODEL)
    vc = tl.load(v_base + (start_n + offs_nc)[:, None] * stride_vk +
                 offs_dc[None, :] * stride_vn)
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
def _softmax_rows_bn64(qk, m_i, sm_scale, m_base, start_n, SUB_M: tl.constexpr,
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
                cols = start_n + tl.arange(0, BLOCK_N)
                row += tl.where((m_base + i) >= cols[None, :], 0.0,
                                float("-inf"))
            qk_scale = al.insert_slice(qk_scale, row, (i, 0), (1, BLOCK_N),
                                       (1, 1))
            m_row = tl.max(row, 1)  # [1]
            m_ij = al.insert_slice(m_ij, m_row, (i, ), (1, ), (1, ))
        m_ij = tl.maximum(m_i, m_ij)
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
def _softmax_rows_bn128(qk, m_i, sm_scale, m_base, start_n,
                        SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        IS_CAUSAL: tl.constexpr):
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
                cols = start_n + tl.arange(0, HALF)
                row_lo += tl.where((m_base + i) >= cols[None, :], 0.0,
                                   float("-inf"))
                row_hi += tl.where((m_base + i) >= (cols + HALF)[None, :], 0.0,
                                   float("-inf"))
            qk_scale = al.insert_slice(qk_scale, row_lo, (i, 0), (1, HALF),
                                       (1, 1))
            qk_scale = al.insert_slice(qk_scale, row_hi, (i, off_hi),
                                       (1, HALF), (1, 1))
            row_max = tl.maximum(row_lo, row_hi)
            row_max_agg = tl.max(row_max, 1)
            tmp_max = al.insert_slice(tmp_max, row_max_agg, (i, ), (1, ),
                                      (1, ))
        m_ij = tl.maximum(m_i, tmp_max)
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
             alpha_scale_b, sm_scale, row0, m_base, start_n, sid,
             SUB_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    if BLOCK_N == 128:
        m_ij, l_ij, p = _softmax_rows_bn128(qk, m_i, sm_scale, m_base, start_n,
                                            SUB_M, BLOCK_N, IS_CAUSAL)
    else:
        m_ij, l_ij, p = _softmax_rows_bn64(qk, m_i, sm_scale, m_base, start_n,
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
                    _pv_matmul(V + qkv_offset, p_l1_0, p_l1_1, pv_ub0, pv_ub1,
                               start_n + batch_idx * BLOCK_N, pvid, stride_vk,
                               stride_vn, BLOCK_M, BLOCK_N, BLOCK_DMODEL)
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
                        start_n + batch_idx * BLOCK_N, sid, SUB_M, BLOCK_N,
                        IS_CAUSAL)
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


attention = _attention.apply
