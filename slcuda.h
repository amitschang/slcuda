#include <cuda.h>
#include <curand.h>

#define CUDA_FORCE 100
#define THREADIDX_X blockIdx.x * blockDim.x + threadIdx.x;
#define THREADIDX_Y blockIdy.y * blockDim.y + threadIdy.y;
#define THREADIDX_Z blockIdz.z * blockDim.z + threadIdz.z;
#define THREADIDX blockIdx.z*(gridDim.x*gridDim.y)+\
  (blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x*gridDim.x+\
  threadIdx.x+blockDim.x*blockIdx.x

int SLCUDA_BLOCK_SIZE;

int SLcuda_Type_Id;
typedef struct
{
  int devid;
  int size;
  int ndims;
  int nelems;
  int valid;
  int type;
  SLindex_Type dims[3];
  void *dptr;
}
SLcuda_Type;

int SLcurand_Type_Id;
typedef struct
{
  curandGenerator_t gen;
}
SLcurand_Type;

void slcuda_compute_dims2d(int N, int bsize, int *dx, int *dy);
SLcuda_Type *slcuda_init_cuda(int size, SLtype type, int ndims, int *dims);
SLcuda_Type *slcuda_pop_cuda(void);
int slcuda_push_cuda(SLcuda_Type *cuda);
