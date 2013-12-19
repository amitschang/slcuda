/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <slang.h>
#include "slcuda.h"

SLANG_MODULE(cudareduce);

__global__ void _cuda_sum_reducef (float *arr, float *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ float data[];

  data[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  data[sid] = arr[x]+arr[x+N];

  if (carry==1 && x==0) data[x] += arr[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    data[sid] += data[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = data[0];
}
__global__ void _cuda_sum_reduced (double *arr, double *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ double ddata[];

  ddata[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  ddata[sid] = arr[x]+arr[x+N];

  if (carry==1 && x==0) ddata[x] += arr[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    ddata[sid] += ddata[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = ddata[0];
}
__global__ void _cuda_sum_reducei (int *arr, int *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ int idata[];

  idata[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  idata[sid] = arr[x]+arr[x+N];

  if (carry==1 && x==0) idata[x] += arr[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    idata[sid] += idata[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = idata[0];
}
/*
 Both the std reduce kernel and the chisqr reduction kernel are mean
 to be called only on the first iteration. All othter calls should be
 to the sum reduce kernel.
*/
__global__ void _cuda_std_reducef (float *arr, float *tmp,
				   float mean, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ float data[];

  data[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  data[sid] = pow(arr[x]-mean,2)+pow(arr[x+N]-mean,2);

  if (carry==1 && x==0) data[0] += pow(arr[2*N]-mean,2);

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    data[sid] += data[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = data[0];
}
__global__ void _cuda_std_reduced (double *arr, double *tmp,
				   double mean, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ double ddata[];

  ddata[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  ddata[sid] = pow(arr[x]-mean,2)+pow(arr[x+N]-mean,2);

  if (carry==1 && x==0) ddata[0] += pow(arr[2*N]-mean,2);

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    ddata[sid] += ddata[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = ddata[0];
}
__global__ void _cuda_std_reducei (int *arr, float *tmp,
				   float mean, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ float data[];

  data[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  data[sid] = pow(arr[x]-mean,2)+pow(arr[x+N]-mean,2);

  if (carry==1 && x==0) data[0] += pow(arr[2*N]-mean,2);

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    data[sid] += data[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = data[0];
}

__global__ void _cuda_chisqr_reducef (float *obs, float *exp,
				     float *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ float data[];

  data[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  data[sid] = pow(obs[x]-exp[x],2)/exp[x]+pow(obs[x+N]-exp[x+N],2)/exp[x+N];

  if (carry==1 && x==0) data[0] += pow(obs[2*N]-exp[2*N],2)/exp[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    data[sid] += data[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = data[0];
}
__global__ void _cuda_chisqr_reduced (double *obs, double *exp,
				     double *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ double ddata[];

  ddata[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  ddata[sid] = pow(obs[x]-exp[x],2)/exp[x]+pow(obs[x+N]-exp[x+N],2)/exp[x+N];

  if (carry==1 && x==0) ddata[0] += pow(obs[2*N]-exp[2*N],2)/exp[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    ddata[sid] += ddata[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = ddata[0];
}
__global__ void _cuda_chisqr_reducei (int *obs, int *exp,
				      float *tmp, int N, int carry)
{
  int x=THREADIDX;
  int stride=blockDim.x/2; //we use 1 dim block
  int sid=threadIdx.x;     //shared mem offset id
  extern __shared__ float data[];

  data[sid] = 0; //for some reason, we have to do here, not if below

  if (x>=N){
    return;
  }

  data[sid] = pow((float)obs[x]-(float)exp[x],2)/(float)exp[x]
    +pow((float)obs[x+N]-(float)exp[x+N],2)/(float)exp[x+N];

  if (carry==1 && x==0)
    data[0] += pow((float)obs[2*N]-(float)exp[2*N],2)/(float)exp[2*N];

  __syncthreads();

  while (stride>0){
    if (sid>=stride){
      return;
    }
    data[sid] += data[sid+stride];
    __syncthreads();
    stride/=2;
  }
  
  if (sid==0)
    tmp[blockIdx.x] = data[0];
}

static double slcuda_reduce_get_value (void *arr, int type){
  int outi;
  float outf;
  double outd;
  switch (type){
  case SLANG_INT_TYPE:
    cudaMemcpy((void *)&outi, arr, SLANG_SIZEOF_INT, cudaMemcpyDeviceToHost);
    outd=(double)outi;
    break;
  case SLANG_FLOAT_TYPE:
    cudaMemcpy((void *)&outf, arr, SLANG_SIZEOF_FLOAT, cudaMemcpyDeviceToHost);
    outd=(double)outf;
    break;
  default:
    cudaMemcpy((void *)&outd, arr, SLANG_SIZEOF_DOUBLE, cudaMemcpyDeviceToHost);
    break;
  }
  return outd;
}

static double slcuda_sum_reduce (SLcuda_Type *arr, int inplace)
{
  int N=arr->nelems;
  void **arg1;
  void **arg2;
  double out;
  void *tmp;
  int dx, dy;
  int dsize = arr->size/arr->nelems;
  int smem = SLCUDA_BLOCK_SIZE*dsize;

  // compute dimensions in case we need temp array
  if (!inplace){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    cudaMalloc((void **)&tmp,dx*dy*dsize);
    arg2 = (void **)&tmp;
  }
  else {
    arg2 = &arr->dptr;
  }
  arg1 = &arr->dptr;
  while (N>1){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    dim3 n_blocks(dx, dy);
    switch (arr->type){
    case SLANG_FLOAT_TYPE:
      _cuda_sum_reducef <<< n_blocks,SLCUDA_BLOCK_SIZE,smem
			>>> ((float *)*arg1,
			     (float *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_DOUBLE_TYPE:
      _cuda_sum_reduced <<< n_blocks,SLCUDA_BLOCK_SIZE,smem
			>>> ((double *)*arg1,
			     (double *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_INT_TYPE:
      _cuda_sum_reducei <<< n_blocks,SLCUDA_BLOCK_SIZE,smem
			>>> ((int *)*arg1,
			     (int *)*arg2,
			     N/2,N%2);
      break;
    }
    arg1 = arg2;
    N = dx*dy;
  }
  out = slcuda_reduce_get_value((void *)*arg2, arr->type);
  if (!inplace)
    cudaFree((void *)tmp);
  return out;
}
static float slcuda_std_reduce (SLcuda_Type *arr, float mean, int inplace)
{
  int N=arr->nelems;
  void **arg1;
  void **arg2;
  float out;
  float *tmp;
  int dx, dy, otype;
  int dsize = arr->size/arr->nelems;
  int smem = SLCUDA_BLOCK_SIZE*dsize;
  // compute dimensions in case we need temp array
  if (!inplace || arr->type == SLANG_INT_TYPE){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    cudaMalloc((void **)&tmp,dx*dy*dsize);
    arg2 = (void **)&tmp;
  }
  else {
    arg2 = &arr->dptr;
  }
  arg1 = &arr->dptr;
  // first reduction is std one, all others are sum reductions
  slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
  dim3 n_blocks(dx, dy);
  switch (arr->type){
  case SLANG_FLOAT_TYPE:
    _cuda_std_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
		      >>> ((float *)*arg1,
			   (float *)*arg2,
			   mean,
			   N/2,N%2);
    otype=SLANG_FLOAT_TYPE;
    break;
  case SLANG_DOUBLE_TYPE:
    _cuda_std_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
		      >>> ((double *)*arg1,
			   (double *)*arg2,
			   mean,
			   N/2,N%2);
    otype=SLANG_DOUBLE_TYPE;
    break;
  case SLANG_INT_TYPE:
    _cuda_std_reducei <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
		      >>> ((int *)*arg1,
			   (float *)*arg2,
			   mean,
			   N/2,N%2);
    otype=SLANG_FLOAT_TYPE;
    break;
  }
  arg1 = arg2;
  N = dx*dy;
  while (N>1){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    dim3 n_blocks(dx, dy);
    switch (arr->type){
    case SLANG_FLOAT_TYPE:
      _cuda_sum_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((float *)*arg1,
			     (float *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_DOUBLE_TYPE:
      _cuda_sum_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((double *)*arg1,
			     (double *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_INT_TYPE:
      _cuda_sum_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((float *)*arg1,
			     (float *)*arg2,
			     N/2,N%2);
      break;
    }      
    N = dx*dy;
  }
  out = slcuda_reduce_get_value((void *)*arg2, otype);
  if (!inplace)
    cudaFree((void *)tmp);
  return out;
}
static float slcuda_chisqr_reduce (SLcuda_Type *obs, SLcuda_Type *exp, int inplace)
{
  int N=obs->nelems;
  void **arg1;
  void **arg2;
  float out;
  float *tmp;
  int dx, dy, otype;
  int dsize = obs->size/obs->nelems;
  int smem = SLCUDA_BLOCK_SIZE*dsize;
  // compute dimensions in case we need temp array
  if (!inplace || obs->type == SLANG_INT_TYPE){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    cudaMalloc((void **)&tmp,dx*dy*dsize);
    arg2 = (void **)&tmp;
  }
  else {
    arg2 = &obs->dptr;
  }
  arg1 = &obs->dptr;
  // first reduction is std one, all others are sum reductions
  slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
  dim3 n_blocks(dx, dy);
  switch (obs->type){
  case SLANG_FLOAT_TYPE:
    _cuda_chisqr_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			 >>> ((float *)*arg1,
			      (float *)exp->dptr,
			      (float *)*arg2,
			      N/2,N%2);
    otype=SLANG_FLOAT_TYPE;
    break;
  case SLANG_DOUBLE_TYPE:
    _cuda_chisqr_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			 >>> ((double *)*arg1,
			      (double *)exp->dptr,
			      (double *)*arg2,
			      N/2,N%2);
    otype=SLANG_DOUBLE_TYPE;
    break;
  case SLANG_INT_TYPE:
    _cuda_chisqr_reducei <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			 >>> ((int *)*arg1,
			      (int *)exp->dptr,
			      (float *)*arg2,
			      N/2,N%2);
    otype=SLANG_FLOAT_TYPE;
    break;
  }
  arg1 = arg2;
  N = dx*dy;
  while (N>1){
    slcuda_compute_dims2d( N/2, SLCUDA_BLOCK_SIZE, &dx, &dy);
    dim3 n_blocks(dx, dy);
    switch (obs->type){
    case SLANG_FLOAT_TYPE:
      _cuda_sum_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((float *)*arg1,
			     (float *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_DOUBLE_TYPE:
      _cuda_sum_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((double *)*arg1,
			     (double *)*arg2,
			     N/2,N%2);
      break;
    case SLANG_INT_TYPE:
      _cuda_sum_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			>>> ((float *)*arg1,
			     (float *)*arg2,
			     N/2,N%2);
      break;
    }      
    N = dx*dy;
  }
  out = slcuda_reduce_get_value((void *)*arg2, otype);
  if (!inplace)
    cudaFree((void *)tmp);
  return out;
}

static void slcuda_sum (void)
{
  SLcuda_Type *cuda;
  double sum;
  int inplace = 0;

  if (SLang_Num_Function_Args==2)
    SLang_pop_int(&inplace);
  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  sum = slcuda_sum_reduce(cuda,inplace);
  SLang_push_double(sum);
}

static void slcuda_mean (void)
{
  SLcuda_Type *cuda;
  float mean;
  int inplace = 0;

  if (SLang_Num_Function_Args==2)
    SLang_pop_int(&inplace);
  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  mean = slcuda_sum_reduce(cuda,0)/(float)cuda->nelems;
  SLang_push_double(mean);
}

static void slcuda_stddev (void)
{
  SLcuda_Type *cuda;
  double mean, sdev;

  if (NULL==(cuda=slcuda_pop_cuda()))
    return;

  mean = slcuda_sum_reduce(cuda,0)/(float)cuda->nelems;
  sdev = sqrt(slcuda_std_reduce(cuda,mean,0)/((float)cuda->nelems-1));
  SLang_push_double(sdev);
}
    
static void slcuda_chisqr (void)
{
  SLcuda_Type *cuda_obs;
  SLcuda_Type *cuda_exp;
  double chisqr;

  if (NULL==(cuda_exp=slcuda_pop_cuda()))
    return;
  if (NULL==(cuda_obs=slcuda_pop_cuda()))
    return;

  if (cuda_exp->nelems != cuda_obs->nelems ||
      cuda_exp->type != cuda_obs->type){
    SLang_verror(SL_USAGE_ERROR, "Arrays must have the same number of elements and type");
  }

  chisqr = slcuda_chisqr_reduce(cuda_obs,cuda_exp,0);
  SLang_push_double(chisqr);
}

static SLang_Intrin_Fun_Type Module_Intrinsics [] =
{
  MAKE_INTRINSIC_0("cusum", slcuda_sum, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cumean", slcuda_mean, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("custd", slcuda_stddev, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("cuchisqr", slcuda_chisqr, SLANG_VOID_TYPE),
  SLANG_END_INTRIN_FUN_TABLE
};

  
int init_cudareduce_module_ns (char *ns_name)
{
   SLang_NameSpace_Type *ns;

   if (NULL == (ns = SLns_create_namespace (ns_name)))
     return -1;

   if ((-1 == SLns_add_intrin_fun_table (ns, Module_Intrinsics, NULL))
       )
     return -1;
   
   return 0;
}

void deinit_cudareduce_module (void)
{
}
