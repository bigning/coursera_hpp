#include "util.h"
#include "cuda_op.h"
#include <cuda.h>
#include <cuda_runtime.h>

void CUDAOp::set_use_gpu(bool use_gpu) {
    if (use_gpu == false) {
        use_gpu_ = false;
        return;
    }
    if (gpu_num_ == 0) {
        std::cout << "sorry, there is no gpu device" << std::endl;
        exit(-1);
    }
    use_gpu_ = true;
}

void CUDAOp::vec_add_cpu(float* p_a, float* p_b, int n, float* p_c) {
    for (int i = 0; i < n ; i++) {
        p_c[i] = p_a[i] + p_b[i];
    }
}

void CUDAOp::vec_add(float* p_a, float* p_b, int n, float* p_c) {
    if (use_gpu_) {
        vec_add_gpu(p_a, p_b, n, p_c);
    }
    else {
        vec_add_cpu(p_a, p_b, n, p_c);
    }
}

CUDAOp::CUDAOp(bool use_gpu) {
    cudaGetDeviceCount(&gpu_num_);
    if (!use_gpu) {
        use_gpu_ = false;
        return;
    }
    if (gpu_num_ == 0) {
        std::cout << "sorry, there is no gpu device" << std::endl;
        exit(-1);
    } 
    use_gpu_ = true;
}

void CUDAOp::test_vec_add(int n) {
    float* h_a = NULL;
    float* h_b = NULL;
    float* h_c_cpu = NULL;
    float* h_c_gpu = NULL;

    h_a = new float[n];
    h_b = new float[n];
    h_c_cpu = new float[n];
    h_c_gpu = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_c_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_c_gpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    
    // firsty use cpu to calculate
    set_use_gpu(false);
    vec_add_cpu(h_a, h_b, n, h_c_cpu);

    // then use gpu to calculate
    set_use_gpu(true);
    vec_add_gpu(h_a, h_b, n, h_c_gpu);

    if (is_two_vec_equal(h_c_cpu, h_c_gpu, n)) {
        std::cout << "[INFO]: cpu result is equal to gpu result!" << std::endl;
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    
}

bool CUDAOp::is_two_vec_equal(float* p_a, float* p_b, int n) {
    bool res = true;
    for (int i = 0; i < n; i++) {
        if (p_a[i] != p_b[i]) {
            std::cout << "[INFO]: first inequality occurs at " << i << ", " << p_a[i] << " != " << p_b[i] << "!" << std::endl;
            return false;
        }
    }
    return true;
}
