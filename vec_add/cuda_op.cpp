#include "util.h"
#include "cuda_op.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

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
    
    clock_t t;
    // firsty use cpu to calculate
    set_use_gpu(false);
    t = clock();
    vec_add_cpu(h_a, h_b, n, h_c_cpu);
    t = clock() - t;
    printf("[INFO]:cpu time: %f s\n", ((float)t) / CLOCKS_PER_SEC);

    // then use gpu to calculate
    set_use_gpu(true);
    t = clock();
    vec_add_gpu(h_a, h_b, n, h_c_gpu);
    t = clock() - t;
    printf("[INFO]:gpu time: %f s\n", ((float)t) / CLOCKS_PER_SEC);

    int diff_index = is_two_vec_equal(h_c_cpu, h_c_gpu, n);
    if (diff_index == -1) {
        std::cout << "[INFO]: cpu result is equal to gpu result!" << std::endl;
    }
    else {
        printf("[INFO]:a[%d](%f) + b[%d](%f) = \n ", diff_index, h_a[diff_index], diff_index, h_b[diff_index]);
        printf("\t\tcpu:%f\n\t\tgpu:%f\n", h_c_cpu[diff_index], h_c_gpu[diff_index]);
    }
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    
}

int CUDAOp::is_two_vec_equal(float* p_a, float* p_b, int n) {
    // returned positived number gives the first index at which the two input vector is not same, return -1 means the two vectors are same
    bool res = true;
    for (int i = 0; i < n; i++) {
        if (p_a[i] != p_b[i]) {
            return i;
        }
    }
    return -1;
}
