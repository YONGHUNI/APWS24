#include <cstdio>

#include "matmul.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;


__global__ void matmul_kernel(float *_A, float *_B, float *_C, int M, int N, int K) {



  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= M || j >= N) return;

  float sum = 0.;

  for (int n = 0; n < K; n++)
  {
    sum += _A[i * K + n] * _B[n * N + j];
  }

  _C[i * N + j] = sum;

}


// void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
//   for (int i = 0; i < M; i++) {
//     for (int k = 0; k < K; k++) {
//       for (int j = 0; j < N; j++) {
//         _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
//       }
//     }
//   }
// }

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  //naive_cpu_matmul(_A, _B, _C, M, N, K);


  // (TODO) Upload A and B matrix to GPU

  cudaMemcpy(A_gpu, _A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, _B, sizeof(float)*K*N, cudaMemcpyHostToDevice);
  
  // (TODO) Launch kernel on a GPU

  
  dim3 blockdim(32, 32);
  dim3 gridDim((M + blockdim.x - 1) / blockdim.x,
               (N + blockdim.y - 1) / blockdim.y);

  matmul_kernel<<<gridDim, blockdim>>>(A_gpu, B_gpu, C_gpu,
                                            M, N, K);

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu,  
                sizeof(float)*M*N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory

  
  CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(float) * M * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.



  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
