#include <chrono>
#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

int main() {
  int bytes = sizeof(int) * (1 << 20);

  int *d_a;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));

  /* 1. Pageable memory test */
  {
    int *a_pageable;
    // TODO: Allocate pageable memory using malloc
    a_pageable = (int *) malloc(bytes);

    auto start = std::chrono::system_clock::now();
    // TODO: Run H2D memcpy on pageable memory
    CHECK_CUDA(cudaMemcpy(d_a, a_pageable, sizeof(bytes), cudaMemcpyHostToDevice));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Pageable memory bandwidth: %lf GB/s\n",
           (bytes / diff.count() / 1e9));
           
    free(a_pageable);
  }

  /* 2. Pinned memory test */
  {
    int *a_pinned;
    // TODO: Allocate pinned memory using cudaMallocHost
    CHECK_CUDA(cudaMallocHost(&a_pinned, bytes));

    auto start = std::chrono::system_clock::now();
    // TODO: Run H2D memcpy on pinned memory
    CHECK_CUDA(cudaMemcpy(d_a, a_pinned, sizeof(bytes), cudaMemcpyHostToDevice));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Pinned memory bandwidth: %lf GB/s\n",
           (bytes / diff.count() / 1e9));
  }

  CHECK_CUDA(cudaMalloc(&d_a, bytes));


  return 0;
}
