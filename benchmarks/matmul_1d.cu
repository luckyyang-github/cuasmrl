// Simple 1D-threaded matmul kernel for RL scheduling demo.
// Each thread computes a single C[i,j].
// Block configuration must be 1D: blockDim = (32 * num_warps, 1, 1)

extern "C" __global__ void matmul_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * N;
  if (tid >= total) return;

  int i = tid / N;  // row in C
  int j = tid % N;  // col in C

  float acc = 0.0f;
  for (int k = 0; k < N; ++k) {
    acc += A[i * N + k] * B[k * N + j];
  }
  C[i * N + j] = acc;
}

