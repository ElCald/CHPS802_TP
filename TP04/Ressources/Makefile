
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

reduction:	reduction.cu Makefile
		nvcc reduction.cu -o reduction $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f reduction

