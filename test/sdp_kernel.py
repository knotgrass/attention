import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# https://chat.openai.com/share/585d3602-3dfb-422f-8fac-99e0416cf994
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
batch_size = 1
nheads = 1
seq_len = 3
head_size = 64
size = (batch_size, nheads, seq_len, head_size)

# Flash attention             supports                  float16
# Memory Efficient attention supports        float32 and float16
# The c++ implementation    supports float64, float32 and float16
factory_kwargs = {'device':'cuda:0',
                  'dtype': torch.bfloat16}

query = torch.rand(size, **factory_kwargs)
key = torch.rand(size, **factory_kwargs)
value = torch.rand(size, **factory_kwargs)

with sdpa_kernel(
    [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ],
    set_priority=True,
):
    out = F.scaled_dot_product_attention(query, key, value)

print(out.shape) # torch.Size([1, 3, 1, 64])
print(out.device) # cuda:0
print(out.dtype)
