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

// non-device image smoothing for comparison
static void _test_smooth (void){
  SLang_Array_Type *arr;
  SLang_Array_Type *ker;
  SLang_Array_Type *aro;
  if (-1==SLang_pop_array(&ker,0)) return;
  if (-1==SLang_pop_array(&arr,0)) return;
  int N = arr->num_elements;
  int stride = arr->dims[0];
  int kx = ker->dims[0];
  int ky = ker->dims[1];
  int i,j,x,ix,iy,idxx,idxy;
  float *img = (float *)arr->data;
  float *kernel = (float *)ker->data;
  float *out;
  out = (float *)malloc(N*sizeof(float));
  for (x=0;x<N;x++){
    ix=x%stride;
    iy=x/stride;
    out[x]=0;
    for (i=0;i<kx;i++){
      for (j=0;j<ky;j++){
	// need to mirror if on edge
	idxx=abs(ix+(i-kx/2));
	idxy=abs(iy+(j-ky/2));
	if (idxx >= stride)   idxx=2*stride-idxx-1;
	if (idxy >= N/stride) idxy=2*N/stride-idxy-1;
	out[x]+=kernel[j*kx+i]*img[idxy*stride+idxx];
      }
    }
  }
  aro = SLang_create_array(SLANG_FLOAT_TYPE, 0, (VOID_STAR)out,
			   arr->dims, arr->num_dims);
  SLang_push_array(aro,1);
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
__global__ void _cuda_std_reducef (float *arr, float *tmp, float mean, int N, int carry)
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
__global__ void _cuda_std_reduced (double *arr, double *tmp, double mean, int N, int carry)
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
__global__ void _cuda_std_reducei (int *arr, float *tmp, float mean, int N, int carry)
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

static float slcuda_sum_reduce (SLcuda_Type *arr, int inplace)
{
  int N=arr->nelems;
  void **arg1;
  void **arg2;
  float out;
  float *tmp;
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
  cudaMemcpy((void *)&out, (void *)*arg2, dsize, cudaMemcpyDeviceToHost);
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
  int dx, dy;
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
    break;
  case SLANG_DOUBLE_TYPE:
    _cuda_std_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
		      >>> ((double *)*arg1,
			   (double *)*arg2,
			   mean,
			   N/2,N%2);
    break;
  case SLANG_INT_TYPE:
    _cuda_std_reducef <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
		      >>> ((float *)*arg1,
			   (float *)*arg2,
			   mean,
			   N/2,N%2);
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
  cudaMemcpy((void *)&out, (void *)*arg2, dsize, cudaMemcpyDeviceToHost);
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
  int dx, dy;
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
    break;
  case SLANG_DOUBLE_TYPE:
    _cuda_chisqr_reduced <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			 >>> ((double *)*arg1,
			      (double *)exp->dptr,
			      (double *)*arg2,
			      N/2,N%2);
    break;
  case SLANG_INT_TYPE:
    _cuda_chisqr_reducei <<< n_blocks, SLCUDA_BLOCK_SIZE, smem
			 >>> ((int *)*arg1,
			      (int *)exp->dptr,
			      (float *)*arg2,
			      N/2,N%2);
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
  cudaMemcpy((void *)&out, (void *)*arg2, sizeof(float), cudaMemcpyDeviceToHost);
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
