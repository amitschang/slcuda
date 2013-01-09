/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <slang.h>
SLANG_MODULE(cuda);

#define CUDA_FORCE 100

int SLCUDA_BLOCK_SIZE=256;

/* 
   define a bunch of kernels to be run using global memory, as assigned
   by the cuda array initialization code 
*/
__global__ void _cuda_init_dev_array (float value, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N)
    output[i]=value;
}  
__global__ void _cuda_add_cuda (float *a, float *b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]+b[i];
  }
}
__global__ void _cuda_sub_cuda (float *a, float *b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]-b[i];
  }
}
__global__ void _cuda_mul_cuda (float *a, float *b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]*b[i];
  }
}
__global__ void _cuda_div_cuda (float *a, float *b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]/b[i];
  }
}
__global__ void _cuda_pow_cuda (float *a, float *b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=pow(a[i],b[i]);
  }
}
__global__ void _cuda_add_scalar (float *a, float b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]+b;
  }
}
__global__ void _cuda_sub_scalar (float *a, float b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]-b;
  }
}
__global__ void _cuda_mul_scalar (float *a, float b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]*b;
  }
}
__global__ void _cuda_div_scalar (float *a, float b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=a[i]/b;
  }
}
__global__ void _cuda_pow_scalar (float *a, float b, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N){
    output[i]=pow(a[i],b);
  }
}
__global__ void _cuda_calc_acc3 (float *p, float *m, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>N)
    return;
  // these are indeces of this threads particle
  int x = i*3; int y = x+1; int z = x+2;
  // variables
  int   iix,iiy,iiz;
  float rx,ry,rz,r,F;

  // initialize output to zero force
  output[x]=0; output[y]=0; output[z]=0;

  // add up force vectors from all other particles
  for (int ii=0; ii<N; ii++){
    if (ii==i) continue;
    iix = 3*ii; iiy = iix+1; iiz = iix+2;
    rx  = p[iix]-p[x]; ry = p[iiy]-p[y]; rz = p[iiz]-p[z];
    r   = sqrt(pow(rx,2)+pow(ry,2)+pow(rz,2));
    F   = m[ii]/r;
    output[x]+=F*(rx/r);
    output[y]+=F*(ry/r);
    output[z]+=F*(rz/r);
  }
}
__global__ void _cuda_calc_acc2 (float *p, float *m, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>N)
    return;
  // these are indeces of this threads particle
  int x = i*2; int y = x+1;
  // variables
  int   iix,iiy;
  float rx,ry,r,F;

  // initialize output to zero force
  output[x]=0; output[y]=0;

  // add up force vectors from all other particles
  for (int ii=0; ii<N; ii++){
    if (ii==i) continue;
    iix = 2*ii; iiy = iix+1;
    rx  = p[iix]-p[x]; ry = p[iiy]-p[y];
    r   = sqrt(pow(rx,2)+pow(ry,2));
    F   = m[ii]/r;
    output[x]+=F*(rx/r);
    output[y]+=F*(ry/r);
  }
}
__global__ void _cuda_calc_acc1 (float *p, float *m, float *output, int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>N)
    return;
  // these are indeces of this threads particle
  // initialize output to zero force
  float rx;

  output[i]=0;

  // add up force vectors from all other particles
  for (int ii=0; ii<N; ii++){
    if (ii==i) continue;
    rx = p[ii]-p[i];
    output[i]+=m[ii]/copysignf(pow(rx,2),rx);
  }
}
__global__ void _cuda_image_smooth (float *img, float *kernel, float *out,
				    int kx, int ky, int stride, int N)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  if (x>N)
    return;

  int imgx=x%stride;
  int imgy=x/stride;
  int idxx,idxy;
  out[x]=0;
  // loop over all items in kernel
  for (int i=0;i<kx;i++){
    for (int j=0;j<ky;j++){
      // need to mirror if on edge
      idxx=abs(imgx+(i-kx/2));
      idxy=abs(imgy+(j-ky/2));
      if (idxx >= stride)   idxx=2*stride-idxx-1;
      if (idxy >= N/stride) idxy=2*N/stride-idxy-1;
      out[x]+=kernel[j*kx+i]*img[idxy*stride+idxx];
    }
  }
}
__global__ void _cuda_vector_smooth (float *img, float *kernel, float *out,
				    int kx, int N)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  if (x>N)
    return;

  int idx;
  out[x]=0;
  // loop over all items in kernel
  for (int i=0;i<kx;i++){
      // need to mirror if on edge
      idx=abs(x+(i-kx/2));
      if (idx >= N) idx=2*N-idx-1;
      out[x]+=kernel[i]*img[idx];
  }
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
  
  if (0==slcuda_check_device_presence())
    {
      SLang_push_int(0);
      return;
    }
  
  SLang_pop_int(&devid);
  if (cudaSuccess==cudaGetDeviceProperties(&devprops,devid))
    SLang_push_cstruct(&devprops,SLcuda_Device_Struct);
}

static void slcuda_meminfo (void)
{
  int devid=0;
  size_t free,total;
  if (SLANG_INT_TYPE==SLang_peek_at_stack())
    SLang_pop_int(&devid);
  cudaMemGetInfo(&free,&total);
  SLang_push_int(free);
  SLang_push_int(total);
}

static int SLcuda_Type_Id = -1;
typedef struct
{
  int devid;
  int size;
  int ndims;
  int nelems;
  int valid;
  SLindex_Type dims[3];
  void *dptr;
}
SLcuda_Type;

static void slcuda_cuda_info (void)
{
  SLcuda_Type *cuda;
  SLang_MMT_Type *mmt;

  if (NULL==(mmt=SLang_pop_mmt(SLcuda_Type_Id)))
    return;

  if (NULL==(cuda=(SLcuda_Type *)SLang_object_from_mmt(mmt)))
    return;

  printf("mmt object at %p\n",mmt);
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

static SLcuda_Type *slcuda_init_cuda(int size, int ndims, int *dims)
{
  float *out;
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
  cuda_o->dptr=(void *)out;
  cuda_o->valid=1;
  return cuda_o;
}
  
static void slcuda_init_array (void)
{
  SLang_Array_Type *arr;
  int size;
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;

  if (-1==SLang_pop_array_of_type(&arr,SLANG_FLOAT_TYPE))
    return;
  
  size=(arr->sizeof_type)*(arr->num_elements);
  cuda=slcuda_init_cuda(size, arr->num_dims, arr->dims);
  cudaMemcpy(cuda->dptr,arr->data,size,cudaMemcpyHostToDevice);

  SLang_free_array(arr);
  mmt=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR) cuda);

  if (0==SLang_push_mmt(mmt))
    return;

  SLang_free_mmt(mmt);
}

static void slcuda_init_dev_array (void)
{
  SLang_Array_Type *dims;
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;
  double initval;
  int initvalspecified=0;
  int nelems=1;

  if (2==SLang_Num_Function_Args){
    SLang_pop_double(&initval);
    initvalspecified=1;
  }
  
  if (-1==SLang_pop_array_of_type(&dims,SLANG_INT_TYPE))
    return;
  
  for (int i=0;i<dims->num_elements;i++)
    nelems=nelems*((int *)dims->data)[i];

  cuda=slcuda_init_cuda(nelems*sizeof(float),dims->num_elements,(int *)dims->data);

  SLang_free_array(dims);

  int n_blocks = nelems/SLCUDA_BLOCK_SIZE;
  if ((n_blocks*nelems)<nelems) n_blocks++;

  if (initvalspecified)
    _cuda_init_dev_array <<< n_blocks, SLCUDA_BLOCK_SIZE >>> ((float)initval,
							      (float *)cuda->dptr,
							      nelems);
  
  mmt=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR) cuda);

  if (0==SLang_push_mmt(mmt)){
    return;
  }
  
  SLang_free_mmt(mmt);
}

static void slcuda_fetch_array (void)
{
  SLang_Array_Type *arr;
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;
  float *darr;

  mmt=SLang_pop_mmt(SLcuda_Type_Id);

  if (NULL==mmt)
    return;

  if (NULL==(cuda=(SLcuda_Type *)SLang_object_from_mmt(mmt)))
    return;

  darr = (float *)SLmalloc(cuda->size);

  cudaMemcpy(darr,cuda->dptr,cuda->size,cudaMemcpyDeviceToHost);

  if (NULL==(arr=SLang_create_array(SLANG_FLOAT_TYPE,0,(VOID_STAR) darr,
				    cuda->dims, cuda->ndims))){
    SLfree((char *)darr);
    return;
  }

  (void) SLang_push_array(arr,1);
}

static void slcuda_smooth (void)
{
  SLang_MMT_Type *mmt_img;
  SLang_MMT_Type *mmt_kernel;
  SLang_MMT_Type *mmt_o;
  SLcuda_Type *cuda_img;
  SLcuda_Type *cuda_kernel;
  SLcuda_Type *cuda_o;

  // if we are given three args, then the output goes to that cuda
  // object
  if (3==SLang_Num_Function_Args){
    if (NULL==(mmt_o=SLang_pop_mmt(SLcuda_Type_Id)))
      return;
    if (NULL==(cuda_o=(SLcuda_Type *)SLang_object_from_mmt(mmt_o)))
      return;
  }
  // get image and kernel
  if (NULL==(mmt_kernel=SLang_pop_mmt(SLcuda_Type_Id)))
    return;
  if (NULL==(mmt_img=SLang_pop_mmt(SLcuda_Type_Id)))
    return;
  if (NULL==(cuda_img=(SLcuda_Type *)SLang_object_from_mmt(mmt_img)))
    return;
  if (NULL==(cuda_kernel=(SLcuda_Type *)SLang_object_from_mmt(mmt_kernel)))
      return;
  // Image and kernel should be no more than 2d and kernel needs to be
  // odd by odd dimensions
  if (0==cuda_kernel->dims[0]%2||
      2<cuda_kernel->ndims||
      2<cuda_img->ndims){
    printf("Wrong dimensions for smoothing\n");
    return;
  }
  if (2==cuda_kernel->ndims&&
      0==cuda_kernel->dims[1]%2){
    printf("kernel 2nd dimention is not odd\n");
    return;
  }

  if (3!=SLang_Num_Function_Args){
    cuda_o=slcuda_init_cuda(cuda_img->size,cuda_img->ndims,cuda_img->dims);
  }

  int n_blocks=cuda_img->nelems/SLCUDA_BLOCK_SIZE;
  if (n_blocks*SLCUDA_BLOCK_SIZE<cuda_img->nelems)
    n_blocks++;

  // should handle 1 or 2 dimensions here
  if (2==cuda_img->ndims)
    _cuda_image_smooth <<< n_blocks, SLCUDA_BLOCK_SIZE >>> ((float *)cuda_img->dptr,
							    (float *)cuda_kernel->dptr,
							    (float *)cuda_o->dptr,
							    cuda_kernel->dims[0],
							    cuda_kernel->dims[1],
							    cuda_img->dims[1],
							    cuda_img->nelems);
  else
    _cuda_vector_smooth <<< n_blocks, SLCUDA_BLOCK_SIZE >>> ((float *)cuda_img->dptr,
							     (float *)cuda_kernel->dptr,
							     (float *)cuda_o->dptr,
							     cuda_kernel->dims[0],
							     cuda_img->nelems);
    

  if (3!=SLang_Num_Function_Args){
    mmt_o=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR) cuda_o);

    if (0==SLang_push_mmt(mmt_o)){
      return;
    }
  
    SLang_free_mmt(mmt_o);
  }
}

static void slcuda_do_binary_op(SLcuda_Type *a, SLcuda_Type *b, SLcuda_Type *o, int op)
{
  void (*func)(float *,float *,float *,int);

  int block_size=SLCUDA_BLOCK_SIZE;
  int n_blocks=a->nelems/block_size;
  if (n_blocks*block_size<a->nelems)
    n_blocks++;
  int N=a->nelems;

  switch (op){
  case SLANG_PLUS:
    func=&_cuda_add_cuda;
    break;
  case SLANG_MINUS:
    func=&_cuda_sub_cuda;
    break;
  case SLANG_TIMES:
    func=&_cuda_mul_cuda;
    break;
  case SLANG_DIVIDE:
    func=&_cuda_mul_cuda;
    break;
  case SLANG_POW:
    func=&_cuda_pow_cuda;
    break;
  case CUDA_FORCE:
    if (a->dims[1]==3)
      func=&_cuda_calc_acc3;
    else if (a->dims[1]==2)
      func=&_cuda_calc_acc2;
    else if (a->dims[1]==1)
      func=&_cuda_calc_acc1;
    n_blocks=(a->nelems/a->ndims)/block_size;
    if (n_blocks*block_size<(a->nelems/a->dims[1]))
      n_blocks++;
    N=a->nelems/a->dims[1];
    break;
  }

  func <<< n_blocks, block_size >>> ((float *)(a->dptr),
				     (float *)(b->dptr),
				     (float *)(o->dptr),
				     N);
}

static void slcuda_do_binary_op_scalar(SLcuda_Type *a, float b, SLcuda_Type *o, int op)
{
  void (*func)(float *, float, float *,int);

  int block_size=SLCUDA_BLOCK_SIZE;
  int n_blocks=a->nelems/block_size;
  if (n_blocks*block_size<a->nelems)
    n_blocks++;

  switch (op){
  case SLANG_PLUS:
    func=&_cuda_add_scalar;
    break;
  case SLANG_MINUS:
    func=&_cuda_sub_scalar;
    break;
  case SLANG_TIMES:
    func=&_cuda_mul_scalar;
    break;
  case SLANG_DIVIDE:
    func=&_cuda_mul_scalar;
    break;
  case SLANG_POW:
    func=&_cuda_pow_scalar;
    break;
  }

  func <<< n_blocks, block_size >>> ((float *)(a->dptr),b,
				     (float *)(o->dptr),
				     a->nelems);
}

static void slcuda_binary_op (int op)
{
  SLang_MMT_Type *mmt_a;
  SLang_MMT_Type *mmt_b;
  SLang_MMT_Type *mmt_o;
  SLcuda_Type *cuda_a;
  SLcuda_Type *cuda_b;
  SLcuda_Type *cuda_o;
  SLtype intype;
  float scalararg;
  int scalar=0;

  // if we are given three args, then the output goes to that cuda
  // object
  if (3==SLang_Num_Function_Args){
    if (NULL==(mmt_o=SLang_pop_mmt(SLcuda_Type_Id)))
      return;
    if (NULL==(cuda_o=(SLcuda_Type *)SLang_object_from_mmt(mmt_o)))
      return;
  }
  // other args can be scalar or cuda type
  intype=SLang_peek_at_stack();
  if (SLANG_INT_TYPE==intype||
      SLANG_FLOAT_TYPE==intype||
      SLANG_DOUBLE_TYPE==intype)
    {
      scalar=1;
      SLang_pop_float(&scalararg);
      if (NULL==(mmt_a=SLang_pop_mmt(SLcuda_Type_Id)))
	return;
    }
  else {
    if (NULL==(mmt_b=SLang_pop_mmt(SLcuda_Type_Id)))
      return;
    
    intype=SLang_peek_at_stack();
    if (SLANG_INT_TYPE==intype||
	SLANG_FLOAT_TYPE==intype||
	SLANG_DOUBLE_TYPE==intype)
      {
	scalar=1;
	SLang_pop_float(&scalararg);
      }
    else {
      if (NULL==(mmt_a=SLang_pop_mmt(SLcuda_Type_Id)))
	return;
    }
  }

  if (NULL==(cuda_a=(SLcuda_Type *)SLang_object_from_mmt(mmt_a)))
    return;

  if (0==scalar){
    if (NULL==(cuda_b=(SLcuda_Type *)SLang_object_from_mmt(mmt_b)))
      return;
  }
  
  if (3!=SLang_Num_Function_Args){
    cuda_o=slcuda_init_cuda(cuda_a->size,cuda_a->ndims,cuda_a->dims);
  }
  if (1==scalar){
    slcuda_do_binary_op_scalar(cuda_a,scalararg,cuda_o,op);
  }
  else {
    slcuda_do_binary_op(cuda_a,cuda_b,cuda_o,op);
  }

  if (3!=SLang_Num_Function_Args){
    mmt_o=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR) cuda_o);

    if (0==SLang_push_mmt(mmt_o)){
      return;
    }
  
    SLang_free_mmt(mmt_o);
  }
}

static void slcuda_add (void)
{
  slcuda_binary_op(SLANG_PLUS);
}

static void slcuda_sub (void)
{
  slcuda_binary_op(SLANG_MINUS);
}

static void slcuda_mul (void)
{
  slcuda_binary_op(SLANG_TIMES);
}

static void slcuda_div (void)
{
  slcuda_binary_op(SLANG_DIVIDE);
}
static void slcuda_pow (void)
{
  slcuda_binary_op(SLANG_POW);
}
static void slcuda_acc (void)
{
  slcuda_binary_op(CUDA_FORCE);
}

static int slcuda_bin_op_result (int op, SLtype a, SLtype b, SLtype *c)
{
  (void) op;
  (void) a;
  (void) b;
  *c = (SLtype)SLcuda_Type_Id;
  return 1;
}

static int slcuda_bin_op_op (int op, SLtype a_type, VOID_STAR ap, SLuindex_Type na,
			     SLtype b_type, VOID_STAR bp, SLuindex_Type nb,
			     VOID_STAR cp)
{
  SLcuda_Type *res;
  SLcuda_Type *in1, *in2;
  SLang_MMT_Type *mmt;
  
  if (NULL==(in1=(SLcuda_Type *)SLang_object_from_mmt(*(SLang_MMT_Type **)ap)))
    return 0;
  if (NULL==(in2=(SLcuda_Type *)SLang_object_from_mmt(*(SLang_MMT_Type **)bp)))
    return 0;

  res=slcuda_init_cuda(in1->size,in1->ndims,in1->dims);

  switch (op){
  case SLANG_PLUS:
    slcuda_do_binary_op(in1,in2,res,SLANG_PLUS);
    break;
  default:
    return 0;
  }

  if (NULL==(mmt=SLang_create_mmt(SLcuda_Type_Id,(VOID_STAR)res)))
    return 0;

  *(SLang_MMT_Type **)cp=mmt;
  return 1;
}

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
  MAKE_INTRINSIC_0("cuadd", slcuda_add, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusub", slcuda_sub, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cumul", slcuda_mul, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cudiv", slcuda_div, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cupow", slcuda_pow, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuacc", slcuda_acc, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusmooth", slcuda_smooth, SLANG_VOID_TYPE),
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

  return 0;
}
  
int init_cuda_module_ns (char *ns_name)
{
   SLang_NameSpace_Type *ns;

   if (NULL == (ns = SLns_create_namespace (ns_name)))
     return -1;

   if (-1 == register_classes ())
     return -1;
   
   if ((-1 == SLns_add_intrin_fun_table (ns, Module_Intrinsics, NULL))
       ||(-1 == SLns_add_intrin_var_table (ns, Module_Variables, NULL))
       )
     return -1;
   
   if (-1 == SLclass_add_binary_op (SLcuda_Type_Id, SLcuda_Type_Id,
				    slcuda_bin_op_op, slcuda_bin_op_result))
     return -1;
   
   return 0;
}

void deinit_cuda_module (void)
{
}
