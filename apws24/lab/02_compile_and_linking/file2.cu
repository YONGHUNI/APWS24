#include <cstdio>

__global__ void welcome_kernel() {
  printf("(Device) Welcome to APSS23!\n");
}

void welcome() {
  printf("(Host) Welcome to APSS23!\n");
  welcome_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}