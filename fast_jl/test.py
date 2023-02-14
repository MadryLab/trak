import torch as ch
import fast_jl
from time import time

B = 32
F = 300_000_000
N = 1024
num_sm = 108

input_data = ch.randn(B, F,device='cuda:0', dtype=ch.float16)

for i in range(1000):
    start = time()
    output = fast_jl.project_rademacher_32(input_data, N, 0, 400)
    ch.cuda.synchronize()
    print(time() - start)
print(output.shape, output.device)


