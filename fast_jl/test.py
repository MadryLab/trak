import torch as ch
import fast_jl

B = 8
F = 300_000_000
N = 1024 * 32
num_sm = 108

input_data = ch.randn(B, F,device='cuda:0', dtype=ch.float16)

output = fast_jl.rademacher(input_data, N, 0)
print(output.shape, output.device)


