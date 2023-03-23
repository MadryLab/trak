#!/usr/bin/env python
from setuptools import setup

setup(name="traker",
      version="0.1.1",
      description="TRAK: Attributing Model Behavior at Scale",
      long_description="Check https://trak.csail.mit.edu/ to learn more about TRAK",
      author="MadryLab",
      author_email='trak@mit.edu',
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
               ],
          'fast':
              ["fast_jl"
              ]},
      include_package_data=True,
      )
