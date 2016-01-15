#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "hello cuda " << device_count << std::endl;
}
