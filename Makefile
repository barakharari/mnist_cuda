CFLAGS= -Wall -Wextra -pedantic
DEBUG= -g


all: nn_parallel.cu
	nvcc $^ -std=c++11 $(DEBUG) -o nn
	
