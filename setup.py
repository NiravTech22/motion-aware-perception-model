import os

print("Setting CUDA_HOME explicitly...")
os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
print(f"CUDA_HOME set to: {os.environ['CUDA_HOME']}")

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the CUDA extension
cuda_ext = CUDAExtension(
    name='accelsight_cuda',
    sources=[
        'src/models/cuda/perception_ops.cpp',
        'src/models/cuda/perception_kernels.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': [
            '-O3',
            '-allow-unsupported-compiler',
            '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
        ]
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
