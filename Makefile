PREFIX=/usr/local
NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -shared
CFLAGS = -fPIC
INC = -I/usr/local/cuda/include
LIB = -L/usr/local/cuda/lib64

default_target: build
all: build install
build:
	$(NVCC) $(NVFLAGS) --compiler-options $(CFLAGS) $(INC) $(LIB) -lslang -lcudart -lcurand cuda-module.cu -o cuda-module.so
install:
	cp cuda-module.so $(PREFIX)/lib/slang/v2/modules/
	cp cuda.sl  $(PREFIX)/share/slsh/local-packages
	cp cuda.hlp $(PREFIX)/share/slsh/local-packages/help
clean:
	rm cuda-module.so
