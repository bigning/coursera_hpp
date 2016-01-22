#include "cuda_op.h"

int main() {
    CUDAOp* cuda_op = new CUDAOp(true);
    cuda_op->test_vec_add();
    return 0;
}
