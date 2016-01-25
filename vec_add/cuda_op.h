#include <iostream>
#define TILE_WIDTH_FOR_MAT_MULTIPLY 16

class CUDAOp {
public:

    CUDAOp(bool use_gpu);
    void set_use_gpu(bool use_gpu);

    // p_c[i] = p_a[i] + p_b[i]
    void vec_add(float* p_a, float* p_b, int n, float* p_c);
    void test_vec_add(int vec_size);

    // A * B = C, where A is a m*k matrix, B is a k*n matrix, C is a m*n matrix
    void matrix_multiply(float* pa, float* pb, float* pc, int m, int n, int k);
    void test_matrix_multiply(int m, int n, int k);
    
private:
    void vec_add_cpu(float* p_a, float* p_b, int n, float* p_c);
    void vec_add_gpu(float* p_a, float* p_b, int n, float* p_c);

    void matrix_multiply_cpu(float* pa, float* pb, float* pc, int m, int n, int k);
    void matrix_multiply_gpu(float* pa, float* pb, float* pc, int m, int n, int k);

    int is_two_vec_equal(float* p_a, float* p_b, int n);

    bool use_gpu_;
    int gpu_num_;
};

