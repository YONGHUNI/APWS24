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

__global__ void pythagoras(int *pa, int *pb, int *pc, int *presult) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *presult = 1;
  else
    *presult = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int result = 0;
  //int *_a,*_b,*_c;
  int *p_da, *p_db, *p_dc, *p_dres;


  // TODO: 1. allocate device memory

  cudaMalloc(&p_da, sizeof(int));
  cudaMalloc(&p_db, sizeof(int));
  cudaMalloc(&p_dc, sizeof(int));
  cudaMalloc(&p_dres, sizeof(int));

  // TODO: 2. copy data to device

  cudaMemcpy(p_da, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_db, &b, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_dc, &c, sizeof(int), cudaMemcpyHostToDevice);


  // TODO: 3. launch kernel
  
  pythagoras<<<1,1>>>(p_da, p_db, p_dc, p_dres);

  // TODO: 4. copy result back to host

  cudaMemcpy(&result, p_dres, sizeof(int), cudaMemcpyDeviceToHost);


  cudaFree(p_da);
  cudaFree(p_db);
  cudaFree(p_dc);
  cudaFree(p_da);


  //END TODO

  if (result) printf("YES\n");
  else printf("NO\n");

  return 0;
}
