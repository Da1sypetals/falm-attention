import torch
from flash_attention import FlashAttentionTriton
from torch.nn.functional import scaled_dot_product_attention as sdpa
import einops as ein


def make_mask(batch_size, seq_len, mask_embed, n_head):
    i = 0
    while True:
        i += 1
        mf = (torch.rand(batch_size, seq_len, mask_embed) < 0.1).to(dtype=torch.float32)
        mask = (ein.einsum(mf, mf, "b l1 d, b l2 d -> b l1 l2") > 0).to(torch.bool)
        mf = mf.cuda()
        mask = mask.cuda()
        if (mask.sum(dim=-1) > 0.1).all():
            break
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
    mf, mask = make_mask(BATCH_SIZE, SEQ_LEN, MASK_DIM, NUM_HEADS)
    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)
    ref_O = sdpa(Q, K, V, mask)
    tri_out = FlashAttentionTriton.apply(Q, K, V, mf, softmax_scale).half()
    diff = ref_O - tri_out
    diffmean = diff.abs().mean() / tri_out.abs().mean()
    print(
        f"SEQ_LEN={SEQ_LEN}: Mean ref = {ref_O.abs().mean()}, Mean diff = {diffmean.item()}"
    )


def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


if __name__ == "__main__":
    # seq_lens = [128, 256, 512, 768, 1024, 1536, 2048]

    primes = [num for num in range(500, 1001) if is_prime(num)]
    # primes = [x * 64 for x in range(30, 50)]
    for seqlen in primes:
        test_op(BATCH_SIZE=1, NUM_HEADS=4, SEQ_LEN=seqlen, MASK_DIM=128, HEAD_DIM=64)
