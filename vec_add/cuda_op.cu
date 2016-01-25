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
    dim3 dim_grid((n - 1) / 256 + 1, 1, 1);
    dim3 dim_block(256, 1, 1);
    vec_add_kernel<<<dim_grid, dim_block>>>(d_a, d_b, d_c, n);

    // copy result from device to host
    copy_device2host(h_c, d_c, size);

    // release device memory
    release_device_mem(d_a);
    release_device_mem(d_b);
    release_device_mem(d_c);
    
    cudaDeviceSynchronize();
}

__global__
void matrix_multiply_kernel(float* pa, float* pb, float* pc, int m, int n, int k) {

    __shared__ float a_block[TILE_WIDTH_FOR_MAT_MULTIPLY][TILE_WIDTH_FOR_MAT_MULTIPLY];
    __shared__ float b_block[TILE_WIDTH_FOR_MAT_MULTIPLY][TILE_WIDTH_FOR_MAT_MULTIPLY];
    int tile_width = TILE_WIDTH_FOR_MAT_MULTIPLY;

    float cvalue = 0.0;

    for (int tile_index = 0; tile_index < (k - 1) / tile_width + 1; tile_index++) {
        // split matrix a and b to tiles
        int row_a = blockIdx.y * blockDim.y + threadIdx.y;
        int col_a = tile_index * tile_width + threadIdx.x;

        if (row_a < m && col_a < k) {
            a_block[threadIdx.y][threadIdx.x] = pa[row_a * k + col_a];
        }
        else {
            a_block[threadIdx.y][threadIdx.x] = 0.0;
        }

        int row_b = tile_index * tile_width + threadIdx.y;
        int col_b = blockDim.x * blockIdx.x + threadIdx.x;
        if (row_b < k && col_b < n) {
            b_block[threadIdx.y][threadIdx.x] = pb[row_b * n + col_b];
        }
        else {
            b_block[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();


        // do tile matrix mutliply
        for (int i = 0; i < tile_width; i++) {
            cvalue += a_block[threadIdx.y][i] * b_block[i][threadIdx.x];
        }
        __syncthreads();
    }

    int row_c = blockIdx.y * blockDim.y + threadIdx.y;
    int col_c = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_c < m && col_c < n) {
        pc[row_c * n + col_c] = cvalue;
    }
}

void CUDAOp::matrix_multiply_gpu(float* pa, float* pb, float* pc, int m, int n, int k) {
    float* da = NULL;
    float* db = NULL;
    float* dc = NULL;

    allocate_and_copy_host2device((void**)&da, pa, m*k*sizeof(float));
    allocate_and_copy_host2device((void**)&db, pb, n*k*sizeof(float));
    allocate_device_mem((void**)&dc, m*n*sizeof(float));

    dim3 grid_size((n - 1) / TILE_WIDTH_FOR_MAT_MULTIPLY + 1, (m - 1) / TILE_WIDTH_FOR_MAT_MULTIPLY + 1, 1);
    dim3 block_size(TILE_WIDTH_FOR_MAT_MULTIPLY, TILE_WIDTH_FOR_MAT_MULTIPLY, 1);
    matrix_multiply_kernel<<<grid_size, block_size>>>(da, db, dc, m, n, k);
    cudaDeviceSynchronize();

    copy_device2host(pc, dc, m*n*sizeof(float));

    release_device_mem(da);
    release_device_mem(db);
    release_device_mem(dc);

}
