from os import environ

environ['TORCH_CUDA_ARCH_LIST']="7.0+PTX"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_jl',
    ext_modules=[
        CUDAExtension('fast_jl', ['fast_jl.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
