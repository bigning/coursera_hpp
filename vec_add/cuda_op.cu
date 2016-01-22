#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_op.h"
#include "util.h"

__global__
void vec_add_kernel(float* pa, float* pb, float* pc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pc[i] = pa[i] + pb[i];
    }
}

void CUDAOp::vec_add_gpu(float* h_a, float* h_b, int n, float* h_c) {
    int size = n * sizeof(float);

    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;

    // allocate and copy data from host to device
    allocate_and_copy_host2device((void**)&d_a, h_a, size);
    allocate_and_copy_host2device((void**)&d_b, h_b, size);
    allocate_device_mem((void**)&d_c, size);

    // kernel code
    dim3 dim_grid((n - 1) / 256, 1, 1);
    dim3 dim_block(256, 1, 1);
    vec_add_kernel<<<dim_grid, dim_block>>>(d_a, d_b, d_c, n);

    
    // copy result from device to host
    copy_device2host(h_c, d_c, size);

    // release device memory
    release_device_mem(d_a);
    release_device_mem(d_b);
    release_device_mem(d_c);
}
