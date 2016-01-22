#include "cuda_op.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    CUDAOp* cuda_op = new CUDAOp(false);
    if (argc == 1) {
        std::cout << "Usage:test vec_add vec_size, i.e. test vec_add 100" << std::endl;
        return 0;
    }
    if (strcmp("vec_add", argv[1]) == 0) { 
        if (argc != 3) {
            std::cout << "Usage:test vec_add vec_size, i.e. test vec_add 100" << std::endl;
            return 0;
        }
        int vec_size = atoi(argv[2]);
        cuda_op->test_vec_add(vec_size);
    }
    else {
        std::cout << "Usage:test vec_add vec_size, i.e. test vec_add 100" << std::endl;
        return 0;
    }
    return 0;
}
