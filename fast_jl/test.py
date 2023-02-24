import torch as ch
import fast_jl
from time import time

B = 16
F = 256
N = 1024
num_sm = 108


input_data = ch.zeros((B, F),device='cuda:0', dtype=ch.float32)
input_data[0, 0] = 1;
input_data[0, 1] = 1;
input_data[0, 2] = 1;
output = fast_jl.project_rademacher_16(input_data, N, 2, 128)
ch.cuda.synchronize()
