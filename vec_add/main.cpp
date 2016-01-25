#include "cuda_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_info() {
    std::cout << "Usage: test" << std::endl;
    std::cout << "\tvec_add vec_size\n";
    std::cout << "\tmatrix_multiply m n k\n";
}

int main(int argc, char* argv[]) {
    CUDAOp* cuda_op = new CUDAOp(false);
    if (argc == 1) {
        print_info();
        return 0;
    }
    if (strcmp("vec_add", argv[1]) == 0) { 
        if (argc != 3) {
            print_info();
            return 0;
        }
        int vec_size = atoi(argv[2]);
        cuda_op->test_vec_add(vec_size);
    }
    else if (strcmp("matrix_multiply", argv[1]) == 0) {
        if (argc != 5) {
            print_info();
            return 0;
        }
        int m = atoi(argv[2]);
        int n = atoi(argv[3]);
        int k = atoi(argv[4]);
        cuda_op->test_matrix_multiply(m, n, k);
    }
    else {
        print_info();
        return 0;
    }
    return 0;
}
