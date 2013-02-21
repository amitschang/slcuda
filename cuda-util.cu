/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <slang.h>
#include "slcuda.h"

SLANG_MODULE(cudautil);

__global__ void _cuda_add_cuda (float *a, long *b, float *output, int N)
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

static void slcuda_do_binary_op(SLcuda_Type *a,
				SLcuda_Type *b,
				SLcuda_Type *o,
				int op)
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
  }

  func <<< n_blocks, block_size >>> ((float *)(a->dptr),
				     (float *)(b->dptr),
				     (float *)(o->dptr),
				     N);
}

static void slcuda_do_binary_op_scalar(SLcuda_Type *a, double b,
				       SLcuda_Type *o, int op)
{
  void (*funcf)(float *, float, float *,int);
  void (*funcd)(double *, double, double *,int);
  void (*funci)(int *, float, float *,int);

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
  double scalararg;
  int scalar=0;

  // if we are given three args, then the output goes to that cuda
  // object
  if (3==SLang_Num_Function_Args){
    if (NULL==(cuda_o=slcuda_pop_cuda()))
      return;
  }
  // other args can be scalar or cuda type
  intype=SLang_peek_at_stack();
  if (intype == SLcuda_Type_Id){
    if (NULL=(cuda_b==slcuda_pop_cuda()))
      return;
  }
  else {
    scalar = 1;
    SLang_pop_double(&scalararg);
  }
  // finally, first arg should be cuda
  if (NULL=(cuda_a==slcuda_pop_cuda()))
    return;

  if (0==scalar){
    if (cuda_a->nelems != cuda_b->nelems ||
	cuda_a->type != cuda_b->type){
      SLang_verror(SL_USAGE_ERROR,"Cuda arrays should be same length & type");
      return;
    }
  }

  if (3!=SLang_Num_Function_Args){
    cuda_o=slcuda_init_cuda(cuda_a->size,SLANG_FLOAT_TYPE,
			    cuda_a->ndims,cuda_a->dims);
  }
  if (1==scalar){
    slcuda_do_binary_op_scalar(cuda_a,scalararg,cuda_o,op);
  }
  else {
    slcuda_do_binary_op(cuda_a,cuda_b,cuda_o,op);
  }

  if (3!=SLang_Num_Function_Args)
    slcuda_push_cuda(cuda_o);
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
  MAKE_INTRINSIC_0("cusum", slcuda_sum, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cumean", slcuda_mean, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("custd", slcuda_stddev, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuchisqr", slcuda_chisqr, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cusmooth", slcuda_smooth, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("testsmooth", _test_smooth, SLANG_VOID_TYPE),
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
