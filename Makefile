NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -shared
CFLAGS = -fPIC
INC = -I/usr/local/cuda/include
LIB = -L/usr/local/cuda/lib64

default_target: all
all:
	$(NVCC) $(NVFLAGS) --compiler-options $(CFLAGS) $(INC) $(LIB) -lslang -lcudart cuda-module.cu -o cuda-module.so
clean:
	rm cuda-module.so
