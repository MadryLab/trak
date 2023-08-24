#!/usr/bin/env python

from os import environ
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

environ['TORCH_CUDA_ARCH_LIST'] = "7.0+PTX"

long_description = open('README.rst').read()

setup(
            name='fast_jl',
                version="0.1.3",
                    description="Fast JL: Compute JL projection fast on a GPU",
                        author="MadryLab",
                            author_email='trak@mit.edu',
                                install_requires=["torch>=2.0.0"],
                                    long_description=long_description,
                                        ext_modules=[
                                                    CUDAExtension(
                                                                    'fast_jl', 
                                                                                sources=['fast_jl.cu'],
                                                                                            library_dirs=[environ['CONDA_PREFIX'] + '/lib'],
                                                                                                        include_dirs=[environ['CONDA_PREFIX'] + '/include'],
                                                                                                                    libraries=['cudart', 'cublas', 'curand']  # Add any other libraries if needed
                                                                                                                            )
                                                        ],
                                            cmdclass={
                                                        'build_ext': BuildExtension
                                                            },
                                                setup_requires=["torch>=2.0.0"]
                                                )

