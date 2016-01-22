PROJECT_NAME = cuda_test.out

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = g++

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

# CFLAGS = -c -m32 -I$(CUDAPATH)/include
# NVCCFLAGS = -c -I$(CUDAPATH)/include
CFLAGS = -c
NVCCFLAGS = -c

# this is mac only. only linux system, remove -Xlinker and add in -lGL -lGLUT?
LFLAGS = -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: gpu

# build: gpu
#	$(NVCC) $(LFLAGS) -o $(PROJECT_NAME) *.cu *.cpp

gpu:
	$(NVCC) *.cu *.cpp -o $(PROJECT_NAME)

cpu:			
	$(CC) $(CFLAGS) *.cpp

clean:
	rm *.out
