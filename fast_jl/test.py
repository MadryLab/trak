import torch as ch
import fast_jl
from time import time

B = 8
F = 256
N = 512
num_sm = 108


for i in range(F):
    input_data = ch.zeros((B, F),device='cuda:0', dtype=ch.float32)
    input_data *= 0
    input_data[0, i] = 1
    output = fast_jl.project_rademacher_32(input_data, N, 0, 128)
    ch.cuda.synchronize()
    if output.sum(1).max() != 1:
        print(i)
        print(output)
        print("BAD")
        break


