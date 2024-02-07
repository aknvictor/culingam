#ifndef CUDA_CODE_CUH
#define CUDA_CODE_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <nvToolsExt.h>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void end_end_residual(double *data, int M, int N, int m, int *U, int uN);
double* causal_order(double *data, int m, int n, double *mlist);

#endif //CUDA_CODE_CUH
