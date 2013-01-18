/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <slang.h>
#include "slcuda.h"

SLANG_MODULE(cudautil);

__global__ void _cuda_add_cuda (float *a, float *b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]+b[i];
  }
}
__global__ void _cuda_sub_cuda (float *a, float *b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]-b[i];
  }
}
__global__ void _cuda_mul_cuda (float *a, float *b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]*b[i];
  }
}
__global__ void _cuda_div_cuda (float *a, float *b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]/b[i];
  }
}
__global__ void _cuda_pow_cuda (float *a, float *b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=pow(a[i],b[i]);
  }
}
__global__ void _cuda_add_scalar (float *a, float b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]+b;
  }
}
__global__ void _cuda_sub_scalar (float *a, float b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]-b;
  }
}
__global__ void _cuda_mul_scalar (float *a, float b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]*b;
  }
}
__global__ void _cuda_div_scalar (float *a, float b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=a[i]/b;
  }
}
__global__ void _cuda_pow_scalar (float *a, float b, float *output, int N)
{
  int i=THREADIDX;
  if (i<N){
    output[i]=pow(a[i],b);
  }
}
__global__ void _cuda_calc_acc3 (float *p, float *m, float *output, int N)
{
  int i=THREADIDX;
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
  int i=THREADIDX;
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
  int i=THREADIDX;
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
  int x=THREADIDX;
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
  int x=THREADIDX;
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

__global__ void _cuda_reduce (float *arr, float *tmp, int carry, int N)
{
  int x=THREADIDX;
  if (x>N-1)
    return;
  tmp[x] = arr[x]+arr[x+N];
  if (carry==1 && x==N-1)
    tmp[x] += arr[2*N];
}

static float slcuda_reduce (SLcuda_Type *arr)
{
  int p=arr->nelems;
  int n=(arr->nelems)/2.0;
  int carry=0;
  int dx, dy;
  float *tmp;
  float out;
  void **arg;
  cudaMalloc((void **) &tmp, n*sizeof(float));
  arg = &arr->dptr;
  while (n>0){
    carry = p - 2*n;
    slcuda_compute_dims2d( n, SLCUDA_BLOCK_SIZE, &dx, &dy);
    dim3 n_blocks(dx, dy);
    _cuda_reduce <<< n_blocks, SLCUDA_BLOCK_SIZE >>> ((float *)*arg,
						      (float *)tmp,
						      carry,
						      n);
    arg = (void **)&tmp;
    p=n;
    n/=2;
  }
  cudaMemcpy((void *)&out, (void *)tmp, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree((void *)tmp);
  return out;
}

static void slcuda_call_reduce (void)
{
  SLang_MMT_Type *mmt;
  SLcuda_Type *cuda;
  float sum;

  if (NULL==(mmt=SLang_pop_mmt(SLcuda_Type_Id)))
    return;
  if (NULL==(cuda=(SLcuda_Type *)SLang_object_from_mmt(mmt)))
    return;
  
  sum = slcuda_reduce(cuda);
  
  SLang_push_double((double) sum);
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
    cuda_o=slcuda_init_cuda(cuda_img->size,SLANG_FLOAT_TYPE,cuda_img->ndims,cuda_img->dims);
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
    cuda_o=slcuda_init_cuda(cuda_a->size,SLANG_FLOAT_TYPE,cuda_a->ndims,cuda_a->dims);
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

static SLang_Intrin_Fun_Type Module_Intrinsics [] =
{
  MAKE_INTRINSIC_0("cuadd", slcuda_add, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusub", slcuda_sub, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cumul", slcuda_mul, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cudiv", slcuda_div, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cupow", slcuda_pow, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuacc", slcuda_acc, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusum", slcuda_call_reduce, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusmooth", slcuda_smooth, SLANG_VOID_TYPE),
  SLANG_END_INTRIN_FUN_TABLE
};

  
int init_cudautil_module_ns (char *ns_name)
{
   SLang_NameSpace_Type *ns;

   if (NULL == (ns = SLns_create_namespace (ns_name)))
     return -1;

   if ((-1 == SLns_add_intrin_fun_table (ns, Module_Intrinsics, NULL))
       )
     return -1;
   
   return 0;
}

void deinit_cudautil_module (void)
{
}
