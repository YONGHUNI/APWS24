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

// Define Tile Size
#define TILE_WIDTH 32

// Matmul Kernel
__global__ void matmul_kernel(float *_A, float *_B, float *_C, int M, int N, int K) {

  //declares two shared memory arrays, which will hold tiles of the input matrices
  __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float sum = 0.0f;


  //calculates the row and column indices of the element that the current thread is responsible for.
  for (int m = 0; m < (TILE_WIDTH + K - 1)/TILE_WIDTH; ++m) {
    if (m*TILE_WIDTH + tx < K && row < M)
      A_s[ty][tx] = _A[row*K + m*TILE_WIDTH + tx];
    else
      A_s[ty][tx] = 0.0;

    if (m*TILE_WIDTH + ty < K && col < N)
      B_s[ty][tx] = _B[(m*TILE_WIDTH + ty)*N + col];
    else
      B_s[ty][tx] = 0.0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k)
      sum += A_s[ty][k] * B_s[k][tx];

    __syncthreads();
  }

  if (row < M && col < N)
    _C[row*N + col] = sum;
}



void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Create CUDA streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Upload A and B matrix to GPU asynchronously
  cudaMemcpyAsync(A_gpu, _A, sizeof(float)*M*K, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(B_gpu, _B, sizeof(float)*K*N, cudaMemcpyHostToDevice, stream2);

  // Launch kernel on a GPU
  dim3 blockdim(32, 32);
  dim3 gridDim((M + blockdim.x - 1) / blockdim.x,
               (N + blockdim.y - 1) / blockdim.y);

  matmul_kernel<<<gridDim, blockdim, 0, stream1>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  // Download C matrix from GPU asynchronously
  CHECK_CUDA(cudaMemcpyAsync(_C, C_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost, stream1));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // Destroy CUDA streams
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
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

