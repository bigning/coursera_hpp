#include <iostream>


class CUDAOp {
public:

    CUDAOp(bool use_gpu);
    void set_use_gpu(bool use_gpu);

    // p_c[i] = p_a[i] + p_b[i]
    void vec_add(float* p_a, float* p_b, int n, float* p_c);
    void test_vec_add(int vec_size);
    
private:
    void vec_add_cpu(float* p_a, float* p_b, int n, float* p_c);
    void vec_add_gpu(float* p_a, float* p_b, int n, float* p_c);
    bool is_two_vec_equal(float* p_a, float* p_b, int n);

    bool use_gpu_;
    int gpu_num_;
};

