#include "util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

void allocate_and_copy_host2device(void** d_address, void* host_address, int size) {
    cudaError_t err;
    err = cudaMalloc(d_address, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }

    err = cudaMemcpy(*d_address, host_address, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
}

void allocate_device_mem(void** d_address, int size) {
    cudaError_t err;
    err = cudaMalloc(d_address, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
}

void copy_device2host(void* h_address, void* d_address, int size) {
    cudaError_t err;
    err = cudaMemcpy(h_address, d_address, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
}

void release_device_mem(void* d_address) {
    cudaError_t err;
    err = cudaFree(d_address);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
}
