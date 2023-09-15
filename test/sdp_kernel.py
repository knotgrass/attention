import torch
import torch.backends.cuda
import torch.nn.functional as F


# https://chat.openai.com/share/585d3602-3dfb-422f-8fac-99e0416cf994
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
batch_size = 1
seq_len = 3
num_head = 1
head_size = 64
size = (batch_size, seq_len, num_head, head_size)

# Flash attention             supports                  float16
# Memory Efficient attention supports        float32 and float16
# The c++ implementation    supports float64, float32 and float16
factory_kwargs = {'device':'cuda:0',
                  'dtype': torch.float16}

query = torch.rand(size, **factory_kwargs)
key = torch.rand(size, **factory_kwargs)
value = torch.rand(size, **factory_kwargs)

with torch.backends.cuda.sdp_kernel(enable_flash= True,
                                    enable_math= False,
                                    enable_mem_efficient = False):
    out = F.scaled_dot_product_attention(query, key, value)

print(out.shape) # torch.Size([1, 3, 1, 64])
print(out.device) # cuda:0
print(out.dtype)
