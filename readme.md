# CULiNGAM - CUDA Accelerated LiNGAM Analysis

## Introduction

CULiNGAM is a high-performance library that accelerates Linear Non-Gaussian Acyclic Model (LiNGAM) analysis on GPUs. It leverages the computing power of NVIDIA GPUs to provide fast and efficient computations for LiNGAM applications, making it ideal for large-scale data analysis in fields such as bioinformatics, economics, and machine learning.

## Installation

To install CULiNGAM, you need a system with NVIDIA CUDA installed and a compatible GPU. This library is tested with CUDA 12.2 and designed for GPUs with the architecture `sm_86`, although it may work with other versions and architectures with appropriate adjustments.

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 8.6 or higher
- CUDA Toolkit 12.2 or compatible version
- Python 3.6 or newer
- Numpy
- A C++17 compatible compiler

### Installing from PyPI

CULiNGAM is available on PyPI and can be easily installed with pip:

```bash
pip install culingam
```

### Installing CULiNGAM manually

1. Clone the repository to your local machine:
```bash
git clone https://github.com/Viktour19/culingam
```

2. Ensure that CUDA_HOME environment variable is set to your CUDA Toolkit installation path. If not, you can set it as follows (example for default CUDA installation path):
```bash
export CUDA_HOME=/usr/local/cuda-12.2
```

3. Optionally, set the GPU_ARCH environment variable to match your GPU architecture if it differs from the default sm_86:
```bash
export GPU_ARCH=sm_xx  # Replace xx with your GPU's compute capability
```

4. Install the library using pip:
```bash
pip install .
```

### Usage
After installation, you can use CULiNGAM in your Python projects to accelerate LiNGAM analysis. Here's a simple example to get started:

https://github.com/Viktour19/culingam/blob/main/examples/basic.py

### Support
For bugs, issues, and feature requests, please submit a report to the repository's issue tracker. Contributions are also welcome.

### Author
Victor Akinwande

### License and Acknowledgment
This project is licensed under the MIT License - see the LICENSE file for details.
A good amount of the code is adapted from sequential implementations of the LiNGAM algorithms present in CULiNGAM: https://github.com/cdt15/lingam.
