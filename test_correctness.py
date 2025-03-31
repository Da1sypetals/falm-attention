import torch
from flash_attention import FlashAttentionTriton
from torch.nn.functional import scaled_dot_product_attention as sdpa
import einops as ein


def make_mask(batch_size, seq_len, mask_embed, n_head):
    print("Make mask start")
    i = 0
    while True:
        i += 1
        mf = (torch.rand(batch_size, seq_len, mask_embed) < 0.1).to(dtype=torch.float32)
        # mf = torch.ones(batch_size, seq_len, seq_len, dtype=torch.int32)
        mask = (ein.einsum(mf, mf, "b l1 d, b l2 d -> b l1 l2") > 0).to(torch.bool)
        mf = mf.cuda()
        mask = mask.cuda()

        if (mask.sum(dim=-1) > 0.1).all():
            print(f"Make mask ok at iteration {i}")
            break
        else:
            print(f"Make mask failed at iteration {i}, retrying...")
            print

    return mf, mask


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, MASK_DIM, dtype=torch.float16):
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

    mf, mask = make_mask(
        BATCH_SIZE,
        SEQ_LEN,
        MASK_DIM,
        NUM_HEADS,
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    ref_O = sdpa(Q, K, V, mask)
    # ref_O.backward(dO)
    # ref_dV, V.grad = V.grad.clone(), None
    # ref_dK, K.grad = K.grad.clone(), None
    # ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = FlashAttentionTriton.apply(Q, K, V, mf, softmax_scale).half()
    # tri_out.backward(dO)
    # tri_dV, V.grad = V.grad.clone(), None
    # tri_dK, K.grad = K.grad.clone(), None
    # tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2

    diff = ref_O - tri_out
    diffmean = diff.abs().mean() / tri_out.abs().mean()

    print(f"=== Mean = {tri_out.abs().mean()}")
    print(f"=== Mean diff = {diffmean.item()}")
    # assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)

    # print("Compilation and check done")

    # fwd_time_naive = do_bench(lambda: sdpa(Q, K, V))
    # bwd_time_naive = do_bench(lambda: sdpa(Q, K, V).backward(dO))

    # fwd_time_flash = do_bench(
    #     lambda: FlashAttentionTriton.apply(Q, K, V, softmax_scale).half()
    # )
    # bwd_time_flash = do_bench(
    #     lambda: FlashAttentionTriton.apply(Q, K, V, softmax_scale).half().backward(dO)
    # )

    # print(f"Fwd time naive: {fwd_time_naive}")
    # print(f"Bwd time naive: {bwd_time_naive}")
    # print(f"Fwd time flash: {fwd_time_flash}")
    # print(f"Bwd time flash: {bwd_time_flash}")
    # print()


if __name__ == "__main__":
    seqlen = 256
    test_op(BATCH_SIZE=1, NUM_HEADS=4, SEQ_LEN=seqlen, MASK_DIM=64, HEAD_DIM=64)
    # print("PASSED")
