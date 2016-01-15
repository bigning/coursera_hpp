#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "hello cuda " << device_count << std::endl;

    if (device_count == 0) {
        return 0;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);

        std::cout << "Device " << i << " name:" << device_prop.name << std::endl;
        std::cout << "\tcomputtational capablities: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "\tglobal memory size:" << device_prop.totalGlobalMem << std::endl;
        std::cout << "\tblock dimension:" << device_prop.maxThreadsDim[0] << " x " << device_prop.maxThreadsDim[1] << std::endl;
    }

    return 1;
}
