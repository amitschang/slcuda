PREFIX=/usr/local
NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -shared -arch sm_10
CFLAGS = -fPIC
INC = -I/usr/local/cuda/include
LIB = -L/usr/local/cuda/lib64

default_target: build
all: build install
build:
	$(NVCC) $(NVFLAGS) --compiler-options $(CFLAGS) $(INC) $(LIB) -lslang -lcudart -lcurand cuda-module.cu -o cuda-module.so
	$(NVCC) $(NVFLAGS) --compiler-options $(CFLAGS) $(INC) $(LIB) -lslang -lcudart cuda-util.cu -o cudautil-module.so
	echo variable NVCC_BIN    =\"$(NVCC)\"\;    > cuda.sl
	echo variable NVCC_FLAGS  =\"$(NVFLAGS)\"\; >> cuda.sl
	echo variable NVCC_C_FLAGS=\"$(CFLAGS)\"\;  >> cuda.sl
	echo variable NVCC_INC    =\"$(INC)\"\;     >> cuda.sl
	echo variable NVCC_LIB    =\"$(LIB)\"\;     >> cuda.sl
	echo                                        >> cuda.sl
	cat cuda.sl.in                              >> cuda.sl
install:
	cp cuda-module.so $(PREFIX)/lib/slang/v2/modules/
	cp cudautil-module.so $(PREFIX)/lib/slang/v2/modules/
	cp cuda.sl  $(PREFIX)/share/slsh/local-packages
	cp cuda.hlp $(PREFIX)/share/slsh/local-packages/help
	cp slcuda.h $(PREFIX)/include/
clean:
	rm *.so cuda.sl
