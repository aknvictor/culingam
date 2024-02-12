from pathlib import Path
from setuptools import setup
from setuptools_cuda_cpp import CUDAExtension, BuildExtension
import os
import numpy

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.2')
cuda_include_dir = os.path.join(cuda_home, 'include')
cuda_lib_dir = os.path.join(cuda_home, 'lib64')
gpu_arch = os.environ.get('GPU_ARCH', 'sm_86')

cuda_ext_path = Path('src/culingam')
cuda_ext = CUDAExtension(
    name='lingam_cuda',
    include_dirs=[cuda_ext_path / 'include', cuda_include_dir, numpy.get_include()],
    sources=[
        cuda_ext_path / 'basic.cu',
        cuda_ext_path / 'basic_wrapper.cpp',
    ],
    libraries=['cudart', 'cudadevrt', 'nvToolsExt'],
    dlink=True,
    dlink_libraries=['cudart', 'cudadevrt', 'nvToolsExt'],
    extra_compile_args={
        'cxx': ['-g','-std=c++17'],
        'nvcc': ['-std=c++17', f'-arch={gpu_arch}'],
    },
)

setup(
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},
    description='CULiNGAM accelerates LiNGAM analysis on GPUs.',
    version='0.0.1',
    author='Victor Akinwande'
)
