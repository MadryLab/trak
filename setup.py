#!/usr/bin/env python

from os import environ
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

environ["TORCH_CUDA_ARCH_LIST"] = "7.0+PTX"

setup(name="traker-fast",
      version="0.1.0",
      description="TRAK: Understanding Model Predictions at Scale",
      author="MadryLab",
      author_email='krisgrg@mit.edu',
      license_files=('LICENSE.txt', ),
      packages=['trak'],
      install_requires=[
       "torch>=1.13",
       "numpy",
       "tqdm",
       ],
      extras_require={
          'tests':
              ["assertpy",
               "torchvision",
               "open_clip_torch",
               "wget",
               "scipy",
               ]},
      ext_modules=[
          CUDAExtension("fast_jl", ["fast_jl/fast_jl.cu"]),
      ],
      include_package_data=True,
      setup_requires=["torch>=1.13"],
      cmdclass={
          'build_ext': BuildExtension
      })
