#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define bSize 8    // length of the blocks

/*---------------------------------------------------------------------------*/

// You can modify the data structure as you want
struct Tensor {

  /* Alloc memory */
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, buf_, n * sizeof(float));
  }

  ~Tensor() {
    if (buf != nullptr)
      free(buf);
  }

  void set_zero() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0;
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf = nullptr;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};

/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

float *character_embedding_g;
float *W_ir0_g, *W_iz0_g, *W_in0_g, *W_ir1_g, *W_iz1_g, *W_in1_g;
float *W_hr0_g, *W_hz0_g, *W_hn0_g, *W_hr1_g, *W_hz1_g, *W_hn1_g;
float *b_ir0_g, *b_iz0_g, *b_in0_g, *b_ir1_g, *b_iz1_g, *b_in1_g;
float *b_hr0_g, *b_hz0_g, *b_hn0_g, *b_hr1_g, *b_hz1_g, *b_hn1_g;
float *W_fc_g, *b_fc_g;
float *rfloats_g;

float *input_g, *emb_out_g;
float *hidden0_g, *hidden1_g;
float *r0_g, *r1_g, *z0_g, *z1_g, *n0_g, *n1_g, *f_g, *char_prob_g;
float *rtmp00_g, *rtmp01_g, *rtmp02_g, *rtmp03_g, *rtmp04_g;
float *rtmp10_g, *rtmp11_g, *rtmp12_g, *rtmp13_g, *rtmp14_g;
float *ztmp00_g, *ztmp01_g, *ztmp02_g, *ztmp03_g, *ztmp04_g;
float *ztmp10_g, *ztmp11_g, *ztmp12_g, *ztmp13_g, *ztmp14_g;
float *ntmp00_g, *ntmp01_g, *ntmp02_g, *ntmp03_g, *ntmp04_g, *ntmp05_g;
float *ntmp10_g, *ntmp11_g, *ntmp12_g, *ntmp13_g, *ntmp14_g, *ntmp15_g;
float *htmp00_g, *htmp01_g, *htmp02_g;
float *htmp10_g, *htmp11_g, *htmp12_g;
float *ftmp0_g;

cudaEvent_t     tick, tock;
cudaStream_t    stream;
cudaGraph_t     graph;
cudaGraphExec_t instance;
float timedelta;  // [ms]
bool graph_created = false;

/* Operations */

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1];
  for (size_t i = 0; i < n; i++) {
    int x = (int)input->buf[0];
    output->buf[i] = weight->buf[x * n + i];
  }
}

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

__global__ void elemwise_add_kernel(float *input1, float *input2, float *output, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  output[i] = input1[i] + input2[i];
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}

__global__ void elemwise_oneminus_kernel(float *input, float *output, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  output[i] = 1.0 - input[i];
}

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}

__global__ void elemwise_mul_kernel(float *input1, float *input2, float *output, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  output[i] = input1[i] * input2[i];
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}

__global__ void elemwise_tanh_kernel(float *input, float *output, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  output[i] = tanh(input[i]);
}

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}

__global__ void elemwise_sigmoid_kernel(float *input, float *output, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  output[i] = 1.0 / (1.0 + expf(-input[i]));
}

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

__global__ void matvec_kernel(float *input1, float *input2, float *output, size_t N_, size_t K_) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  float c = 0.0;
  if (i >= N_) return;
  for (size_t j = 0; j < K_; j++) {
    c += input1[i * K_ + j] * input2[j];
  }
  output[i] = c;
}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, char *parameter_fname) {

  /* Only the root process reads the parameter */
 
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *)read_binary(parameter_fname, &parameter_binary_size);

  /* Network parameters */
  character_embedding =
      new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

  W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
  W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
  W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
  W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
  W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
  W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

  W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
  W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
  W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
  W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
  W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
  W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

  b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
  b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
  b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
  b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
  b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
  b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

  b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
  b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
  b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
  b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
  b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
  b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

  W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
  b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);

  /* input, activations, output, etc. */
  input = new Tensor({1});
  emb_out = new Tensor({EMBEDDING_DIM});

  hidden0 = new Tensor({HIDDEN_DIM});
  hidden1 = new Tensor({HIDDEN_DIM});

  r0 = new Tensor({HIDDEN_DIM});
  r1 = new Tensor({HIDDEN_DIM});
  z0 = new Tensor({HIDDEN_DIM});
  z1 = new Tensor({HIDDEN_DIM});
  n0 = new Tensor({HIDDEN_DIM});
  n1 = new Tensor({HIDDEN_DIM});
  f = new Tensor({NUM_CHAR});

  rtmp00 = new Tensor({HIDDEN_DIM});
  rtmp01 = new Tensor({HIDDEN_DIM});
  rtmp02 = new Tensor({HIDDEN_DIM});
  rtmp03 = new Tensor({HIDDEN_DIM});
  rtmp04 = new Tensor({HIDDEN_DIM});
  rtmp10 = new Tensor({HIDDEN_DIM});
  rtmp11 = new Tensor({HIDDEN_DIM});
  rtmp12 = new Tensor({HIDDEN_DIM});
  rtmp13 = new Tensor({HIDDEN_DIM});
  rtmp14 = new Tensor({HIDDEN_DIM});

  ztmp00 = new Tensor({HIDDEN_DIM});
  ztmp01 = new Tensor({HIDDEN_DIM});
  ztmp02 = new Tensor({HIDDEN_DIM});
  ztmp03 = new Tensor({HIDDEN_DIM});
  ztmp04 = new Tensor({HIDDEN_DIM});
  ztmp10 = new Tensor({HIDDEN_DIM});
  ztmp11 = new Tensor({HIDDEN_DIM});
  ztmp12 = new Tensor({HIDDEN_DIM});
  ztmp13 = new Tensor({HIDDEN_DIM});
  ztmp14 = new Tensor({HIDDEN_DIM});

  ntmp00 = new Tensor({HIDDEN_DIM});
  ntmp01 = new Tensor({HIDDEN_DIM});
  ntmp02 = new Tensor({HIDDEN_DIM});
  ntmp03 = new Tensor({HIDDEN_DIM});
  ntmp04 = new Tensor({HIDDEN_DIM});
  ntmp05 = new Tensor({HIDDEN_DIM});
  ntmp10 = new Tensor({HIDDEN_DIM});
  ntmp11 = new Tensor({HIDDEN_DIM});
  ntmp12 = new Tensor({HIDDEN_DIM});
  ntmp13 = new Tensor({HIDDEN_DIM});
  ntmp14 = new Tensor({HIDDEN_DIM});
  ntmp15 = new Tensor({HIDDEN_DIM});

  htmp00 = new Tensor({HIDDEN_DIM});
  htmp01 = new Tensor({HIDDEN_DIM});
  htmp02 = new Tensor({HIDDEN_DIM});
  htmp10 = new Tensor({HIDDEN_DIM});
  htmp11 = new Tensor({HIDDEN_DIM});
  htmp12 = new Tensor({HIDDEN_DIM});

  rfloats = new Tensor({N * MAX_LEN});
  ftmp0 = new Tensor({NUM_CHAR});
  char_prob = new Tensor({NUM_CHAR});

  // Allocate device memory.
  CHECK_CUDA(cudaMalloc(&W_ir0_g, sizeof(float)*(W_ir0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_iz0_g, sizeof(float)*(W_iz0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_in0_g, sizeof(float)*(W_in0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_ir1_g, sizeof(float)*(W_ir1->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_iz1_g, sizeof(float)*(W_iz1->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_in1_g, sizeof(float)*(W_in1->num_elem())));

  CHECK_CUDA(cudaMalloc(&W_hr0_g, sizeof(float)*(W_hr0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_hz0_g, sizeof(float)*(W_hz0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_hn0_g, sizeof(float)*(W_hn0->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_hr1_g, sizeof(float)*(W_hr1->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_hz1_g, sizeof(float)*(W_hz1->num_elem())));
  CHECK_CUDA(cudaMalloc(&W_hn1_g, sizeof(float)*(W_hn1->num_elem())));

  CHECK_CUDA(cudaMalloc(&b_ir0_g, sizeof(float)*(b_ir0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_iz0_g, sizeof(float)*(b_iz0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_in0_g, sizeof(float)*(b_in0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_ir1_g, sizeof(float)*(b_ir1->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_iz1_g, sizeof(float)*(b_iz1->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_in1_g, sizeof(float)*(b_in1->num_elem())));

  CHECK_CUDA(cudaMalloc(&b_hr0_g, sizeof(float)*(b_hr0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_hz0_g, sizeof(float)*(b_hz0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_hn0_g, sizeof(float)*(b_hn0->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_hr1_g, sizeof(float)*(b_hr1->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_hz1_g, sizeof(float)*(b_hz1->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_hn1_g, sizeof(float)*(b_hn1->num_elem())));

  CHECK_CUDA(cudaMalloc(&W_fc_g, sizeof(float)*(W_fc->num_elem())));
  CHECK_CUDA(cudaMalloc(&b_fc_g, sizeof(float)*(b_fc->num_elem())));

  CHECK_CUDA(cudaMalloc(&emb_out_g, sizeof(float)*(emb_out->num_elem())));

  CHECK_CUDA(cudaMalloc(&hidden0_g, sizeof(float)*(hidden0->num_elem())));
  CHECK_CUDA(cudaMalloc(&hidden1_g, sizeof(float)*(hidden1->num_elem())));

  CHECK_CUDA(cudaMalloc(&r0_g, sizeof(float)*(r0->num_elem())));
  CHECK_CUDA(cudaMalloc(&r1_g, sizeof(float)*(r1->num_elem())));
  CHECK_CUDA(cudaMalloc(&z0_g, sizeof(float)*(z0->num_elem())));
  CHECK_CUDA(cudaMalloc(&z1_g, sizeof(float)*(z1->num_elem())));
  CHECK_CUDA(cudaMalloc(&n0_g, sizeof(float)*(n0->num_elem())));
  CHECK_CUDA(cudaMalloc(&n1_g, sizeof(float)*(n1->num_elem())));
  CHECK_CUDA(cudaMalloc(&f_g, sizeof(float)*(f->num_elem())));

  CHECK_CUDA(cudaMalloc(&rtmp00_g, sizeof(float)*(rtmp00->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp01_g, sizeof(float)*(rtmp01->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp02_g, sizeof(float)*(rtmp02->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp03_g, sizeof(float)*(rtmp03->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp04_g, sizeof(float)*(rtmp04->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp10_g, sizeof(float)*(rtmp10->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp11_g, sizeof(float)*(rtmp11->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp12_g, sizeof(float)*(rtmp12->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp13_g, sizeof(float)*(rtmp13->num_elem())));
  CHECK_CUDA(cudaMalloc(&rtmp14_g, sizeof(float)*(rtmp14->num_elem())));

  CHECK_CUDA(cudaMalloc(&ztmp00_g, sizeof(float)*(ztmp00->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp01_g, sizeof(float)*(ztmp01->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp02_g, sizeof(float)*(ztmp02->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp03_g, sizeof(float)*(ztmp03->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp04_g, sizeof(float)*(ztmp04->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp10_g, sizeof(float)*(ztmp10->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp11_g, sizeof(float)*(ztmp11->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp12_g, sizeof(float)*(ztmp12->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp13_g, sizeof(float)*(ztmp13->num_elem())));
  CHECK_CUDA(cudaMalloc(&ztmp14_g, sizeof(float)*(ztmp14->num_elem())));

  CHECK_CUDA(cudaMalloc(&ntmp00_g, sizeof(float)*(ntmp00->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp01_g, sizeof(float)*(ntmp01->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp02_g, sizeof(float)*(ntmp02->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp03_g, sizeof(float)*(ntmp03->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp04_g, sizeof(float)*(ntmp04->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp05_g, sizeof(float)*(ntmp04->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp10_g, sizeof(float)*(ntmp10->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp11_g, sizeof(float)*(ntmp11->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp12_g, sizeof(float)*(ntmp12->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp13_g, sizeof(float)*(ntmp13->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp14_g, sizeof(float)*(ntmp14->num_elem())));
  CHECK_CUDA(cudaMalloc(&ntmp15_g, sizeof(float)*(ntmp15->num_elem())));

  CHECK_CUDA(cudaMalloc(&htmp00_g, sizeof(float)*(htmp00->num_elem())));
  CHECK_CUDA(cudaMalloc(&htmp01_g, sizeof(float)*(htmp01->num_elem())));
  CHECK_CUDA(cudaMalloc(&htmp02_g, sizeof(float)*(htmp02->num_elem())));
  CHECK_CUDA(cudaMalloc(&htmp10_g, sizeof(float)*(htmp10->num_elem())));
  CHECK_CUDA(cudaMalloc(&htmp11_g, sizeof(float)*(htmp11->num_elem())));
  CHECK_CUDA(cudaMalloc(&htmp12_g, sizeof(float)*(htmp12->num_elem())));

  CHECK_CUDA(cudaMalloc(&ftmp0_g, sizeof(float)*(ftmp0->num_elem())));

  // Initialize CUDA streams and events.
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&tick));
  CHECK_CUDA(cudaEventCreate(&tock));
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  // Upload the network parameters to the GPU.
  // CHECK_CUDA(cudaEventRecord(tick));
  CHECK_CUDA(cudaMemcpy(W_ir0_g, W_ir0->buf, sizeof(float)*W_ir0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_iz0_g, W_iz0->buf, sizeof(float)*W_iz0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_in0_g, W_in0->buf, sizeof(float)*W_in0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_ir1_g, W_ir1->buf, sizeof(float)*W_ir1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_iz1_g, W_iz1->buf, sizeof(float)*W_iz1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_in1_g, W_in1->buf, sizeof(float)*W_in1->num_elem(), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMemcpy(W_hr0_g, W_hr0->buf, sizeof(float)*W_hr0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_hz0_g, W_hz0->buf, sizeof(float)*W_hz0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_hn0_g, W_hn0->buf, sizeof(float)*W_hn0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_hr1_g, W_hr1->buf, sizeof(float)*W_hr1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_hz1_g, W_hz1->buf, sizeof(float)*W_hz1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_hn1_g, W_hn1->buf, sizeof(float)*W_hn1->num_elem(), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(b_ir0_g, b_ir0->buf, sizeof(float)*b_ir0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_iz0_g, b_iz0->buf, sizeof(float)*b_iz0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_in0_g, b_in0->buf, sizeof(float)*b_in0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_ir1_g, b_ir1->buf, sizeof(float)*b_ir1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_iz1_g, b_iz1->buf, sizeof(float)*b_iz1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_in1_g, b_in1->buf, sizeof(float)*b_in1->num_elem(), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(b_hr0_g, b_hr0->buf, sizeof(float)*b_hr0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_hz0_g, b_hz0->buf, sizeof(float)*b_hz0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_hn0_g, b_hn0->buf, sizeof(float)*b_hn0->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_hr1_g, b_hr1->buf, sizeof(float)*b_hr1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_hz1_g, b_hz1->buf, sizeof(float)*b_hz1->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_hn1_g, b_hn1->buf, sizeof(float)*b_hn1->num_elem(), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(W_fc_g, W_fc->buf, sizeof(float)*W_fc->num_elem(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_fc_g, b_fc->buf, sizeof(float)*b_fc->num_elem(), cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaEventRecord(tock));
  // CHECK_CUDA(cudaEventSynchronize(tock));
  // CHECK_CUDA(cudaEventElapsedTime(&timedelta, tick, tock));
  // printf("\n\t(device) upload: %f [ms]\n", timedelta);

  /*--------------------------------------------------------------------------*/
  // Develop the CUDA graph here.
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  /* First layer r */
  matvec_kernel<<<(W_ir0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_ir0_g, emb_out_g, rtmp00_g, W_ir0->shape[0], W_ir0->shape[1]);
  matvec_kernel<<<(W_hr0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hr0_g, hidden0_g, rtmp01_g, W_hr0->shape[0], W_hr0->shape[1]);
  elemwise_add_kernel<<<(rtmp00->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp00_g, b_ir0_g, rtmp02_g, rtmp00->num_elem());
  elemwise_add_kernel<<<(rtmp02->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp02_g, rtmp01_g, rtmp03_g, rtmp02->num_elem());
  elemwise_add_kernel<<<(rtmp03->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp03_g, b_hr0_g, rtmp04_g, rtmp03->num_elem());
  elemwise_sigmoid_kernel<<<(rtmp04->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp04_g, r0_g, rtmp04->num_elem());
  /* First layer z */
  matvec_kernel<<<(W_iz0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_iz0_g, emb_out_g, ztmp00_g, W_iz0->shape[0], W_iz0->shape[1]);
  matvec_kernel<<<(W_hz0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hz0_g, hidden0_g, ztmp01_g, W_hz0->shape[0], W_hz0->shape[1]);
  elemwise_add_kernel<<<(ztmp00->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp00_g, b_iz0_g, ztmp02_g, ztmp00->num_elem());
  elemwise_add_kernel<<<(ztmp02->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp02_g, ztmp01_g, ztmp03_g, ztmp02->num_elem());
  elemwise_add_kernel<<<(ztmp03->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp03_g, b_hz0_g, ztmp04_g, ztmp03->num_elem());
  elemwise_sigmoid_kernel<<<(ztmp04->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp04_g, z0_g, ztmp04->num_elem());
  /* First layer n */
  matvec_kernel<<<(W_in0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_in0_g, emb_out_g, ntmp00_g, W_in0->shape[0], W_in0->shape[1]);
  elemwise_add_kernel<<<(ntmp00->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp00_g, b_in0_g, ntmp01_g, ntmp00->num_elem());
  matvec_kernel<<<(W_hn0->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hn0_g, hidden0_g, ntmp02_g, W_hn0->shape[0], W_hn0->shape[1]);
  elemwise_add_kernel<<<(ntmp02->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp02_g, b_hn0_g, ntmp03_g, ntmp02->num_elem());
  elemwise_mul_kernel<<<(r0->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(r0_g, ntmp03_g, ntmp04_g, r0->num_elem());
  elemwise_add_kernel<<<(ntmp01->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp01_g, ntmp04_g, ntmp05_g, ntmp01->num_elem());
  elemwise_tanh_kernel<<<(ntmp05->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp05_g, n0_g, ntmp05->num_elem());
  /* First layer h (hidden) */
  elemwise_oneminus_kernel<<<(z0->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(z0_g, htmp00_g, z0->num_elem());
  elemwise_mul_kernel<<<(htmp00->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(htmp00_g, n0_g, htmp01_g, htmp00->num_elem());
  elemwise_mul_kernel<<<(z0->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(z0_g, hidden0_g, htmp02_g, z0->num_elem());
  elemwise_add_kernel<<<(htmp01->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(htmp01_g, htmp02_g, hidden0_g, htmp01->num_elem());
  /* Second layer r */
  matvec_kernel<<<(W_ir1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_ir1_g, hidden0_g, rtmp10_g, W_ir1->shape[0], W_ir1->shape[1]);
  matvec_kernel<<<(W_hr1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hr1_g, hidden1_g, rtmp11_g, W_hr1->shape[0], W_hr1->shape[1]);
  elemwise_add_kernel<<<(rtmp10->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp10_g, b_ir1_g, rtmp12_g, rtmp10->num_elem());
  elemwise_add_kernel<<<(rtmp12->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp12_g, rtmp11_g, rtmp13_g, rtmp12->num_elem());
  elemwise_add_kernel<<<(rtmp13->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp13_g, b_hr1_g, rtmp14_g, rtmp13->num_elem());
  elemwise_sigmoid_kernel<<<(rtmp14->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(rtmp14_g, r1_g, rtmp14->num_elem());
  /* Second layer z */
  matvec_kernel<<<(W_iz1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_iz1_g, hidden0_g, ztmp10_g, W_iz1->shape[0], W_iz1->shape[1]);
  matvec_kernel<<<(W_hz1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hz1_g, hidden1_g, ztmp11_g, W_hz1->shape[0], W_hz1->shape[1]);
  elemwise_add_kernel<<<(ztmp10->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp10_g, b_iz1_g, ztmp12_g, ztmp10->num_elem());
  elemwise_add_kernel<<<(ztmp12->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp12_g, ztmp11_g, ztmp13_g, ztmp12->num_elem());
  elemwise_add_kernel<<<(ztmp13->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp13_g, b_hz1_g, ztmp14_g, ztmp13->num_elem());
  elemwise_sigmoid_kernel<<<(ztmp14->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ztmp14_g, z1_g, ztmp14->num_elem());
  /* Second layer n */
  matvec_kernel<<<(W_in1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_in1_g, hidden0_g, ntmp10_g, W_in1->shape[0], W_in1->shape[1]);
  elemwise_add_kernel<<<(ntmp10->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp10_g, b_in1_g, ntmp11_g, ntmp10->num_elem());
  matvec_kernel<<<(W_hn1->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_hn1_g, hidden1_g, ntmp12_g, W_hn1->shape[0], W_hn1->shape[1]);
  elemwise_add_kernel<<<(ntmp12->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp12_g, b_hn1_g, ntmp13_g, ntmp12->num_elem());
  elemwise_mul_kernel<<<(r1->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(r1_g, ntmp13_g, ntmp14_g, r1->num_elem());
  elemwise_add_kernel<<<(ntmp11->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp11_g, ntmp14_g, ntmp15_g, ntmp11->num_elem());
  elemwise_tanh_kernel<<<(ntmp15->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ntmp15_g, n1_g, ntmp15->num_elem());
  /* Second layer h (hidden) */
  elemwise_oneminus_kernel<<<(z1->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(z1_g, htmp10_g, z1->num_elem());
  elemwise_mul_kernel<<<(htmp10->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(htmp10_g, n1_g, htmp11_g, htmp10->num_elem());
  elemwise_mul_kernel<<<(z1->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(z1_g, hidden1_g, htmp12_g, z1->num_elem());
  elemwise_add_kernel<<<(htmp11->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(htmp11_g, htmp12_g, hidden1_g, htmp11->num_elem());
  /* Fully connected layer */
  matvec_kernel<<<(W_fc->shape[0]+bSize-1)/bSize, bSize, 0, stream>>>(W_fc_g, hidden1_g, ftmp0_g, W_fc->shape[0], W_fc->shape[1]);
  elemwise_add_kernel<<<(ftmp0->num_elem()+bSize-1)/bSize, bSize, 0, stream>>>(ftmp0_g, b_fc_g, f_g, ftmp0->num_elem());
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  /*--------------------------------------------------------------------------*/

  // TODO: I think the for loop below should be executed asynchronously so as to
  // use the full capability of SMs in the device.

  /* Generate N names */
  for (int n = 0; n < N; n++) {
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[0] = SOS;
    hidden0->set_zero();
    hidden1->set_zero();
    CHECK_CUDA(cudaMemcpy(hidden0_g, hidden0->buf, sizeof(float)*(hidden0->num_elem()), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(hidden1_g, hidden1->buf, sizeof(float)*(hidden1->num_elem()), cudaMemcpyHostToDevice));

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      embedding(input, character_embedding, emb_out);
      cudaMemcpy(emb_out_g, emb_out->buf, sizeof(float)*emb_out->num_elem(), cudaMemcpyHostToDevice);

      // some calculations replaced by the CUDA graph
      cudaGraphLaunch(instance, stream);
      cudaStreamSynchronize(stream);
 
      /* Softmax */
      cudaMemcpy(f->buf, f_g, sizeof(float)*(f->num_elem()), cudaMemcpyDeviceToHost);
      softmax(f, char_prob);

      /* Random select */
      int selected_char = random_select(char_prob, rfloats, n * MAX_LEN + l);

      output[n * (MAX_LEN + 1) + l] = selected_char;
      input->buf[0] = selected_char;

      if (selected_char == EOS)
        break;
    }
  }
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {

  delete character_embedding;
  delete W_ir0;
  delete W_iz0;
  delete W_in0;
  delete W_ir1;
  delete W_iz1;
  delete W_in1;
  delete W_hr0;
  delete W_hz0;
  delete W_hn0;
  delete W_hr1;
  delete W_hz1;
  delete W_hn1;
  delete b_ir0;
  delete b_iz0;
  delete b_in0;
  delete b_ir1;
  delete b_iz1;
  delete b_in1;
  delete b_hr0;
  delete b_hz0;
  delete b_hn0;
  delete b_hr1;
  delete b_hz1;
  delete b_hn1;
  delete W_fc;
  delete b_fc;
  delete rfloats;

  delete input;
  delete emb_out;
  delete hidden0;
  delete hidden1;
  delete r0;
  delete r1;
  delete z0;
  delete z1;
  delete n0;
  delete n1;
  delete f;
  delete char_prob;
  delete rtmp00;
  delete rtmp01;
  delete rtmp02;
  delete rtmp03;
  delete rtmp04;
  delete rtmp10;
  delete rtmp11;
  delete rtmp12;
  delete rtmp13;
  delete rtmp14;
  delete ztmp00;
  delete ztmp01;
  delete ztmp02;
  delete ztmp03;
  delete ztmp04;
  delete ztmp10;
  delete ztmp11;
  delete ztmp12;
  delete ztmp13;
  delete ztmp14;
  delete ntmp00;
  delete ntmp01;
  delete ntmp02;
  delete ntmp03;
  delete ntmp04;
  delete ntmp05;
  delete ntmp10;
  delete ntmp11;
  delete ntmp12;
  delete ntmp13;
  delete ntmp14;
  delete ntmp15;
  delete htmp00;
  delete htmp01;
  delete htmp02;
  delete htmp10;
  delete htmp11;
  delete htmp12;
  delete ftmp0;
}
