import torch
from triton.testing import do_bench
from flash_attention import FlashAttentionTriton
from torch.nn.functional import scaled_dot_product_attention as sdpa


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    ref_O = sdpa(Q, K, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = FlashAttentionTriton.apply(Q, K, V, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2

    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

    print("Compilation and check done")

    fwd_time_naive = do_bench(lambda: sdpa(Q, K, V))
    bwd_time_naive = do_bench(lambda: sdpa(Q, K, V).backward(dO))

    fwd_time_flash = do_bench(
        lambda: FlashAttentionTriton.apply(Q, K, V, softmax_scale).half()
    )
    bwd_time_flash = do_bench(
        lambda: FlashAttentionTriton.apply(Q, K, V, softmax_scale).half().backward(dO)
    )

    print(f"Fwd time naive: {fwd_time_naive}")
    print(f"Bwd time naive: {bwd_time_naive}")
    print(f"Fwd time flash: {fwd_time_flash}")
    print(f"Bwd time flash: {bwd_time_flash}")
    print()


if __name__ == "__main__":
    seqlen = 256
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=seqlen, HEAD_DIM=64)
    print("PASSED")
