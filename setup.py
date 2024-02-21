from setuptools import setup, find_packages
from setuptools_cuda_cpp import CUDAExtension, BuildExtension
import os
import numpy
from pathlib import Path

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.2')
cuda_include_dir = os.path.join(cuda_home, 'include')
cuda_lib_dir = os.path.join(cuda_home, 'lib64')
gpu_arch = os.environ.get('GPU_ARCH', 'sm_86')

def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ['--threads', '4']

cuda_ext_path = Path('src/culingam')
cuda_ext = CUDAExtension(
    name='lingam_cuda',
    include_dirs=[cuda_ext_path / 'include', cuda_include_dir, numpy.get_include()],
    sources=[
        str(cuda_ext_path / 'basic.cu'),
        str(cuda_ext_path / 'basic_wrapper.cpp'),
    ],
    libraries=['cudart', 'cudadevrt', 'nvToolsExt'],
    extra_compile_args={
        'cxx': ['-g', '-std=c++17'],
        'nvcc': append_nvcc_threads([
            '-O3',
            '-std=c++17',
            f'-arch={gpu_arch}'
        ])
    },
)

setup(
    name='culingam',
    version='0.0.4',
    author='Victor Akinwande',
    description='CULiNGAM accelerates LiNGAM analysis on GPUs.',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension}
)
