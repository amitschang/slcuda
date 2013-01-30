/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <slang.h>
SLANG_MODULE(cuda);

#define SLCURAND_DEFAULT 0
#define SLCURAND_UNIFORM 1
#define SLCURAND_NORMAL 2
#define SLCURAND_LOGNORMAL 3
#define SLCURAND_POISSON 4
#define THREADIDX_X blockIdx.x * blockDim.x + threadIdx.x;
#define THREADIDX_Y blockIdy.y * blockDim.y + threadIdy.y;
#define THREADIDX_Z blockIdz.z * blockDim.z + threadIdz.z;
#define THREADIDX blockIdx.z*(gridDim.x*gridDim.y)+\
  (blockIdx.y*blockDim.y+threadIdx.y)*blockDim.x*gridDim.x+\
  threadIdx.x+blockDim.x*blockIdx.x

static long SLCUDA_MAX_GRID_DIM=65535; //default based on compute arch
static int  SLCUDA_NO_DOUBLE=1;        //which is low. realtime query
int SLCUDA_BLOCK_SIZE=256;

int SLcuda_Type_Id = -1;
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

static int SLcurand_Type_Id = -1;
typedef struct
{
  curandGenerator_t gen;
}
SLcurand_Type;


void slcuda_compute_dims2d(int N, int bsize, int *dx, int *dy)
{
  int _dy, _dx;
  int nb = ceil((float)N/(float)bsize);
  if (nb < SLCUDA_MAX_GRID_DIM){
    *dx = nb;
    *dy = 1;
    return;
  }
  _dx = nb;
  _dy = 1;
  while (_dx > SLCUDA_MAX_GRID_DIM){
    _dy+=1;
    _dx = nb/_dy;
  }
  if (_dx*_dy<nb) _dx++;
  *dy = _dy;
  *dx = _dx;
}

SLcuda_Type *slcuda_pop_cuda(void){
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;
  if (NULL==(mmt=SLang_pop_mmt(SLcuda_Type_Id)))
    return NULL;
  if (NULL==(cuda=(SLcuda_Type *)SLang_object_from_mmt(mmt)))
    return NULL;
  return cuda;
}

int slcuda_push_cuda(SLcuda_Type *cuda){
  SLang_MMT_Type *mmt;
  mmt=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR) cuda);
  if (0==SLang_push_mmt(mmt))
    return 0;
  SLang_free_mmt(mmt);
  return 1;
}

/* 
   Define global kernels for initialization of arrays. These will also
   support range arrays. For single value arrays simply set the init
   to the value and inc to 0
*/
__global__ void _cuda_init_dev_arrayi (int init, int inc,
				       int *output, int N)
{
  int i=THREADIDX;
  if (i<N)
    output[i]=init+i*inc;
}  
__global__ void _cuda_init_dev_arrayf (float init, float inc,
				       float *output, int N)
{
  int i=THREADIDX;
  if (i<N)
    output[i]=init+i*inc;
}  
__global__ void _cuda_init_dev_arrayd (double init, double inc,
				       double *output, int N)
{
  int i=THREADIDX;
  if (i<N)
    output[i]=init+i*inc;
}  
/* 
   Now the S-Lang interaction code
 */

static int slcuda_check_device_presence (void)
{
  int devcount;
  if (cudaSuccess!=cudaGetDeviceCount(&devcount))
    return 0;
  return 1;
}

// How do we make struct fields from char arrays or int arrays???
static SLang_CStruct_Field_Type SLcuda_Device_Struct [] =
  {
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, asyncEngineCount,"asyncEngineCount", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, canMapHostMemory,"canMapHostMemory", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, clockRate,"clockRate", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, computeMode,"computeMode", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, concurrentKernels,"concurrentKernels", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, deviceOverlap,"deviceOverlap", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, ECCEnabled,"ECCEnabled", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, integrated,"integrated", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, kernelExecTimeoutEnabled,"kernelExecTimeoutEnabled", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, l2CacheSize,"l2CacheSize", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, major,"major", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxGridSize,"maxGridSize", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurface1D,"maxSurface1D", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurface1DLayered,"maxSurface1DLayered", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurface2D,"maxSurface2D", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurface2DLayered,"maxSurface2DLayered", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurface3D,"maxSurface3D", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurfaceCubemap,"maxSurfaceCubemap", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxSurfaceCubemapLayered,"maxSurfaceCubemapLayered", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture1D,"maxTexture1D", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture1DLayered,"maxTexture1DLayered", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture1DLinear,"maxTexture1DLinear", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture2D,"maxTexture2D", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture2DGather,"maxTexture2DGather", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture2DLayered,"maxTexture2DLayered", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture2DLinear,"maxTexture2DLinear", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTexture3D,"maxTexture3D", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTextureCubemap,"maxTextureCubemap", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxTextureCubemapLayered,"maxTextureCubemapLayered", 0),
    //MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxThreadsDim,"maxThreadsDim", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxThreadsPerBlock,"maxThreadsPerBlock", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, maxThreadsPerMultiProcessor,"maxThreadsPerMultiProcessor", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, memoryBusWidth,"memoryBusWidth", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, memoryClockRate,"memoryClockRate", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, memPitch,"memPitch", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, minor,"minor", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, multiProcessorCount,"multiProcessorCount", 0),
    //MAKE_CSTRUCT_FIELD(struct cudaDeviceProp, name, "name", SLANG_ARRAY_TYPE, 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, pciBusID,"pciBusID", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, pciDeviceID,"pciDeviceID", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, pciDomainID,"pciDomainID", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, regsPerBlock,"regsPerBlock", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, sharedMemPerBlock,"sharedMemPerBlock", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, surfaceAlignment,"surfaceAlignment", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, tccDriver,"tccDriver", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, textureAlignment,"textureAlignment", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, texturePitchAlignment,"texturePitchAlignment", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, totalConstMem,"totalConstMem", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, totalGlobalMem,"totalGlobalMem", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, unifiedAddressing,"unifiedAddressing", 0),
    MAKE_CSTRUCT_INT_FIELD(struct cudaDeviceProp, warpSize,"warpSize", 0),
    SLANG_END_CSTRUCT_TABLE
  };


static void slcuda_device_info (void)
{
  int devid=0;
  cudaDeviceProp devprops;
  if (1==SLang_Num_Function_Args)
    SLang_pop_int(&devid);
  if (cudaSuccess==cudaGetDeviceProperties(&devprops,devid))
    SLang_push_cstruct(&devprops,SLcuda_Device_Struct);
}

static int slcuda_get_compute_capability (int devid){
  cudaDeviceProp devprops;
  if (cudaSuccess==cudaGetDeviceProperties(&devprops,devid))
    return 10*(10*(&devprops)->major+(&devprops)->minor);
  return -1;
}

static void slcuda_meminfo (void)
{
  int devid=0;
  size_t free,total;
  if (1==SLang_Num_Function_Args)
    SLang_pop_int(&devid);
  cudaMemGetInfo(&free,&total);
  SLang_push_int(free);
  SLang_push_int(total);
}

static void slcuda_cuda_info (void)
{
  SLcuda_Type *cuda;

  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  printf("cuda object at %p\n",cuda);

  if (0==cuda->valid){
    printf("Cuda object invalid (no longer in device memory)\n");
    return;
  }

  printf("Cuda object is on device %d\n",cuda->devid);
  printf("with a total size of %d\n",cuda->size);
  printf("it has %d elemens in %d dimensions\n",cuda->nelems,cuda->ndims);
}  

static void slcuda_free (SLcuda_Type *cuda)
{
  if (0==cuda->valid){
    SLfree((char *)cuda);
    return;
  }
  cudaFree(cuda->dptr);
  SLfree((char *)cuda);
}
    
static void slcuda_destroy_cuda (SLtype type, VOID_STAR f)
{
  SLcuda_Type *cuda;
  cuda=(SLcuda_Type *)f;
  slcuda_free(cuda);
}

static void slcuda_free_intrin (void)
{
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;

  mmt=SLang_pop_mmt(SLcuda_Type_Id);

  if (NULL==mmt)
    return;

  if (NULL==(cuda=(SLcuda_Type *)SLang_object_from_mmt(mmt))){
    SLang_free_mmt(mmt);
    return;
  }
  
  cudaFree(cuda->dptr);
  cuda->valid=0;
}

SLcuda_Type *slcuda_init_cuda(int size, SLtype type, int ndims, int *dims)
{
  void *out;
  SLcuda_Type *cuda_o=
    (SLcuda_Type *)SLmalloc(sizeof(SLcuda_Type));
  memset((char *)cuda_o,0,sizeof(SLcuda_Type));
  cuda_o->devid=0;
  cuda_o->size=size;
  cuda_o->ndims=ndims;
  cuda_o->dims[0]=dims[0];
  cuda_o->dims[1]=dims[1];
  cuda_o->dims[2]=dims[2];
  cuda_o->nelems=dims[0];
  if (ndims>1)
    cuda_o->nelems=(cuda_o->nelems)*dims[1]; 
  if (ndims>2)
    cuda_o->nelems=(cuda_o->nelems)*dims[2];
  // malloc array on device
  cudaMalloc((void **) &out, size);
  cuda_o->dptr=out;
  cuda_o->valid=1;
  cuda_o->type = type;
  return cuda_o;
}
  
static void slcuda_init_array (void)
{
  SLang_Array_Type *arr;
  SLcuda_Type *cuda;
  int size;

  if (-1==SLang_pop_array(&arr,0))
    return;

  if (SLCUDA_NO_DOUBLE &&
      arr->data_type == SLANG_DOUBLE_TYPE){
    SLang_verror(SL_USAGE_ERROR, "double not supported on this GPU");
    return;
  }
  
  size=(arr->sizeof_type)*(arr->num_elements);
  cuda=slcuda_init_cuda(size, arr->data_type, arr->num_dims, arr->dims);
  cudaMemcpy(cuda->dptr,arr->data,size,cudaMemcpyHostToDevice);

  SLang_free_array(arr);
  slcuda_push_cuda(cuda);
}

static void slcuda_init_dev_array (void)
{
  SLang_Array_Type *dims;
  SLcuda_Type *cuda;
  SLtype type;
  double initval;
  double inc=0;
  int initvalspecified=0;
  int nelems=1;
  int size;

  if (4==SLang_Num_Function_Args){
    SLang_pop_double(&inc);
    SLang_pop_double(&initval);
    initvalspecified=1;
    SLang_pop_datatype(&type);
  }    
  else if (3==SLang_Num_Function_Args){
    SLang_pop_double(&initval);
    initvalspecified=1;
    SLang_pop_datatype(&type);
  }
  else if (2==SLang_Num_Function_Args){
    SLang_pop_datatype(&type);
    if (SLCUDA_NO_DOUBLE &&
	type == SLANG_DOUBLE_TYPE){
      SLang_verror(SL_USAGE_ERROR, "double not supported on this GPU");
      return;
    }
  }
  else {
    type=SLANG_FLOAT_TYPE;
  }

  if (-1==SLang_pop_array_of_type(&dims,SLANG_INT_TYPE))
    return;
  
  for (int i=0;i<dims->num_elements;i++)
    nelems=nelems*((int *)dims->data)[i];

  switch (type){
  case SLANG_INT_TYPE:    size=sizeof(int); break;
  case SLANG_FLOAT_TYPE:  size=sizeof(float); break;
  case SLANG_DOUBLE_TYPE: size=sizeof(double); break;
  }

  cuda=slcuda_init_cuda(nelems*size,type,dims->num_elements,(int *)dims->data);

  SLang_free_array(dims);


  if (initvalspecified){
    int bdx, bdy;
    slcuda_compute_dims2d( nelems, SLCUDA_BLOCK_SIZE, &bdx, &bdy);
    dim3 n_blocks(bdx, bdy);

    switch (type){
    case SLANG_INT_TYPE:
      _cuda_init_dev_arrayi <<< 
	n_blocks, SLCUDA_BLOCK_SIZE
			    >>> ((int)initval, (int)inc,
				 (int *)cuda->dptr,
				 nelems);
      break;
    case SLANG_FLOAT_TYPE:
      _cuda_init_dev_arrayf <<< 
	n_blocks, SLCUDA_BLOCK_SIZE
			    >>> ((float)initval,(float)inc,
				 (float *)cuda->dptr,
				 nelems);
      break;
    case SLANG_DOUBLE_TYPE:
      _cuda_init_dev_arrayd <<< 
	n_blocks, SLCUDA_BLOCK_SIZE
			    >>> ((double)initval,(double)inc,
				 (double *)cuda->dptr,
				 nelems);
      break;
    }
  }

  slcuda_push_cuda(cuda);
}

static void slcuda_fetch_array (void)
{
  SLang_Array_Type *arr;
  SLcuda_Type *cuda;
  void *darr;


  if (2==SLang_Num_Function_Args)
    if (-1==SLang_pop_array(&arr,0))
      return;
  
  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  if (2==SLang_Num_Function_Args){
    if (arr->data_type != cuda->type ||
	arr->num_elements != cuda->nelems){
      SLang_push_int(0);
      SLang_free_array(arr);
      return;
    }
    else {
      darr = arr->data;
    }
  }
  else {
    darr = SLmalloc(cuda->size);
  }

  cudaMemcpy(darr,cuda->dptr,cuda->size,cudaMemcpyDeviceToHost);

  if (2==SLang_Num_Function_Args){
    SLang_push_int(1);
    SLang_free_array(arr);
    return;
  }

  if (NULL==(arr=SLang_create_array(cuda->type,0,(VOID_STAR) darr,
				    cuda->dims, cuda->ndims))){
    SLfree((char *)darr);
    return;
  }

  (void) SLang_push_array(arr,1);
}

static long slcurand_calc_seed (SLang_Array_Type *seed_arr)
{
  unsigned long seed;
  int i;

  if (1==seed_arr->num_elements)
    seed = (unsigned long)((int *)seed_arr->data)[0];
  else
    seed = ((unsigned long)((int *)seed_arr->data)[0]<<32) | 
      (unsigned long)((int*)seed_arr->data)[1];
  if (seed_arr->num_elements > 2){
    for (i=2;i<seed_arr->num_elements;i++){
      seed = seed*(unsigned long)((int *)seed_arr->data)[i];
    }
  }
  
  return seed;
}

static void slcurand_new (void)
{
  int type;
  SLang_Array_Type *seed_arr;
  SLang_MMT_Type *mmt_g;
  SLcurand_Type *gen_o;
  curandGenerator_t gen;
  unsigned long seed;
  
  if (2==SLang_Num_Function_Args)
    if (-1==SLang_pop_array_of_type(&seed_arr,SLANG_INT_TYPE))
      return;

  if (-1==SLang_pop_int(&type))
    return;
  
  curandCreateGenerator(&gen, (curandRngType_t)type);

  if (2==SLang_Num_Function_Args){
    seed = slcurand_calc_seed(seed_arr);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
  }
  
  gen_o = (SLcurand_Type *)SLmalloc(sizeof(SLcurand_Type));
  gen_o->gen = gen;

  mmt_g=SLang_create_mmt(SLcurand_Type_Id, (VOID_STAR)gen_o);

  if (0==SLang_push_mmt(mmt_g)){
    return;
  }
  
  SLang_free_mmt(mmt_g);
}

static void slcurand_destroy (SLtype type, VOID_STAR f)
{
  curandDestroyGenerator(((SLcurand_Type *)f)->gen);
}

static void slcurand_seed (void)
{
  SLang_Array_Type *seed_arr;
  SLang_MMT_Type *mmt_g;
  SLcurand_Type *gen;
  unsigned long seed;
  
  if (-1==SLang_pop_array_of_type(&seed_arr,SLANG_INT_TYPE))
    return;

  if (NULL==(mmt_g=SLang_pop_mmt(SLcurand_Type_Id)))
    return;
  if (NULL==(gen=(SLcurand_Type *)SLang_object_from_mmt(mmt_g)))
    return;

  seed = slcurand_calc_seed(seed_arr);

  curandSetPseudoRandomGeneratorSeed(gen->gen, seed);
}

static void slcurand_generate (void)
{
  SLang_MMT_Type *mmt_g;
  SLcurand_Type *gen;
  SLcuda_Type *cuda;
  int type;
  double arg1,arg2;
  curandStatus_t ret;
  
  if (5==SLang_Num_Function_Args){
    if (-1==SLang_pop_double(&arg2))
      return;
    if (-1==SLang_pop_double(&arg1))
      return;
  }
  if (4==SLang_Num_Function_Args){
    if (-1==SLang_pop_double(&arg1))
      return;
  }
  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  if (NULL==(mmt_g=SLang_pop_mmt(SLcurand_Type_Id)))
    return;
  if (NULL==(gen=(SLcurand_Type *)SLang_object_from_mmt(mmt_g)))
    return;

  if (-1==SLang_pop_int(&type))
    return;

  switch(type){
  case SLCURAND_DEFAULT:
    ret = curandGenerate(gen->gen, (unsigned int *)cuda->dptr, cuda->nelems);
    break;
  case SLCURAND_UNIFORM:
    if (cuda->type == SLANG_DOUBLE_TYPE)
      ret = curandGenerateUniformDouble(gen->gen, (double *)cuda->dptr, cuda->nelems);
    else
      ret = curandGenerateUniform(gen->gen, (float *)cuda->dptr, cuda->nelems);
    break;
  case SLCURAND_NORMAL:
    if (cuda->type == SLANG_DOUBLE_TYPE)
      ret = curandGenerateNormalDouble(gen->gen, (double *)cuda->dptr, cuda->nelems,
				       (double)arg1, (double)arg2);
    else
      ret = curandGenerateNormal(gen->gen, (float *)cuda->dptr, cuda->nelems,
				 (float)arg1, (float)arg2);
    break;
  case SLCURAND_LOGNORMAL:
    if (cuda->type == SLANG_DOUBLE_TYPE)
      ret = curandGenerateLogNormalDouble(gen->gen, (double *)cuda->dptr, cuda->nelems,
					  (double)arg1, (double)arg2);
    else
      ret = curandGenerateLogNormal(gen->gen, (float *)cuda->dptr, cuda->nelems,
				    (float)arg1, (float)arg2);
    break;
  }
  SLang_push_int(ret);
}

static SLang_IConstant_Type Module_IConstants [] =
{  
  MAKE_ICONSTANT("CURAND_RNG_PSEUDO_DEFAULT",CURAND_RNG_PSEUDO_DEFAULT),
  MAKE_ICONSTANT("CURAND_RNG_PSEUDO_XORWOW",CURAND_RNG_PSEUDO_XORWOW),
  MAKE_ICONSTANT("CURAND_RNG_PSEUDO_MRG32K3A",CURAND_RNG_PSEUDO_MRG32K3A),
  MAKE_ICONSTANT("CURAND_RNG_PSEUDO_MTGP32",CURAND_RNG_PSEUDO_MTGP32),
  MAKE_ICONSTANT("CURAND_RNG_QUASI_DEFAULT",CURAND_RNG_QUASI_DEFAULT),
  MAKE_ICONSTANT("CURAND_RNG_QUASI_SOBOL32",CURAND_RNG_QUASI_SOBOL32),
  MAKE_ICONSTANT("CURAND_DEFAULT",SLCURAND_DEFAULT),
  MAKE_ICONSTANT("CURAND_UNIFORM",SLCURAND_UNIFORM),
  MAKE_ICONSTANT("CURAND_NORMAL",SLCURAND_NORMAL),
  MAKE_ICONSTANT("CURAND_LOGNORMAL",SLCURAND_LOGNORMAL),
  MAKE_ICONSTANT("CURAND_POISSON",SLCURAND_POISSON),
  SLANG_END_ICONST_TABLE
};

static SLang_Intrin_Var_Type Module_Variables [] =
{
  MAKE_VARIABLE("CUDA_BLOCK_SIZE",(VOID_STAR)&SLCUDA_BLOCK_SIZE, SLANG_INT_TYPE, 0),
  SLANG_END_INTRIN_VAR_TABLE
};

static SLang_Intrin_Fun_Type Module_Intrinsics [] =
{
  MAKE_INTRINSIC_0("cuda_info", slcuda_device_info, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuda_meminfo", slcuda_meminfo, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuda_objinfo", slcuda_cuda_info, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cunew", slcuda_init_array, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuarr", slcuda_init_dev_array, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cufree", slcuda_free_intrin, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuget", slcuda_fetch_array, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("curand_new", slcurand_new, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("curand_seed", slcurand_seed, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("curand_gen", slcurand_generate, SLANG_VOID_TYPE),
  SLANG_END_INTRIN_FUN_TABLE
};

static int register_classes (void)
{
  SLang_Class_Type *cl;

  if (SLcuda_Type_Id != -1)
    return 0;

  if (NULL == (cl = SLclass_allocate_class ("SLcuda_Type")))
    return -1;

  (void) SLclass_set_destroy_function (cl, slcuda_destroy_cuda);

  if (-1 == SLclass_register_class (cl, SLANG_VOID_TYPE,
				    sizeof (SLcuda_Type),
				    SLANG_CLASS_TYPE_MMT))
    return -1;

  SLcuda_Type_Id = SLclass_get_class_id (cl);

  if (NULL == (cl = SLclass_allocate_class ("SLcurand_Type")))
    return -1;

  (void) SLclass_set_destroy_function (cl, slcurand_destroy);
  
  if (-1 == SLclass_register_class (cl, SLANG_VOID_TYPE,
				    sizeof (SLcurand_Type),
				    SLANG_CLASS_TYPE_MMT))
    return -1;

  SLcurand_Type_Id = SLclass_get_class_id (cl);

  return 0;
}
  
int init_cuda_module_ns (char *ns_name)
{
   SLang_NameSpace_Type *ns;

   if (!slcuda_check_device_presence())
     return -1;

   int compcap=slcuda_get_compute_capability(0); // of default device
   if (compcap >= 130)
     SLCUDA_NO_DOUBLE=0; //above 130 double is supported
   if (compcap >= 300)
     SLCUDA_MAX_GRID_DIM=4294967295; //kepler supports large grid sizes

   if (NULL == (ns = SLns_create_namespace (ns_name)))
     return -1;

   if (-1 == register_classes ())
     return -1;
   
   if ((-1 == SLns_add_intrin_fun_table (ns, Module_Intrinsics, NULL))
       ||(-1 == SLns_add_intrin_var_table (ns, Module_Variables, NULL))
       ||(-1 == SLns_add_iconstant_table (ns, Module_IConstants, NULL))
       )
     return -1;
   
   return 0;
}

void deinit_cuda_module (void)
{
}
