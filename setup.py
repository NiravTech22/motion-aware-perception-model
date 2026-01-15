from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Define the CUDA extension
cuda_ext = CUDAExtension(
    name='accelsight_cuda',
    sources=[
        'src/models/cuda/perception_ops.cpp',
        'src/models/cuda/perception_kernels.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3']
    }
)

setup(
    name='accelsight',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[cuda_ext],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
    ],
)
