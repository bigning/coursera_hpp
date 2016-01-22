#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

void allocate_and_copy_host2device(void** d_address, void* host_address, int n);
void allocate_device_mem(void** d_address, int size);
void release_device_mem(void* d_address);
void copy_device2host(void* h_address, void* d_address, int size);
