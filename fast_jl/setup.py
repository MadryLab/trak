from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_jl',
    ext_modules=[
        CUDAExtension('fast_jl', [
            'fast_jl.cpp',
            'fast_jl_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
