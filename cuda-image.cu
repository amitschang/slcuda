/* -*- mode: c -*- */

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <slang.h>
#include "slcuda.h"

SLANG_MODULE(cudaimage);

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

static void slcuda_smooth (void)
{
  SLcuda_Type *cuda_img;
  SLcuda_Type *cuda_kernel;
  SLcuda_Type *cuda_o;

  // if we are given three args, then the output goes to that cuda
  // object
  if (3==SLang_Num_Function_Args)
    if (NULL==(cuda_o=slcuda_pop_cuda()))
      return;
  // get image and kernel
  if (NULL==(cuda_kernel=slcuda_pop_cuda()))
    return;
  if (NULL==(cuda_img=slcuda_pop_cuda()))
    return;

  // Image and kernel should be no more than 2d and kernel needs to be
  // odd by odd dimensions
  if (0==cuda_kernel->dims[0]%2||
      2<cuda_kernel->ndims||
      2<cuda_img->ndims){
    printf("Wrong kernel/image dimensions for smoothing\n");
    return;
  }
  if (2==cuda_kernel->ndims&&
      0==cuda_kernel->dims[1]%2){
    printf("kernel 2nd dimension must be odd\n");
    return;
  }
  // if we are not given a device array to fill, make new one to
  // return
  if (3!=SLang_Num_Function_Args){
    cuda_o=slcuda_init_cuda(cuda_img->size,
			    SLANG_FLOAT_TYPE,
			    cuda_img->ndims,
			    cuda_img->dims);
  }
  
  int dx, dy;
  slcuda_compute_dims2d(cuda_img->nelems, SLCUDA_BLOCK_SIZE, &dx, &dy);
  dim3 n_blocks(dx, dy);

  // should handle 1 or 2 dimensions here
  if (2==cuda_img->ndims)
    _cuda_image_smooth <<<
      n_blocks, SLCUDA_BLOCK_SIZE
		       >>> ((float *)cuda_img->dptr,
			    (float *)cuda_kernel->dptr,
			    (float *)cuda_o->dptr,
			    cuda_kernel->dims[0],
			    cuda_kernel->dims[1],
			    cuda_img->dims[1],
			    cuda_img->nelems);
  else
    _cuda_vector_smooth <<<
      n_blocks, SLCUDA_BLOCK_SIZE
			>>> ((float *)cuda_img->dptr,
			     (float *)cuda_kernel->dptr,
			     (float *)cuda_o->dptr,
			     cuda_kernel->dims[0],
			     cuda_img->nelems);
    

  if (3!=SLang_Num_Function_Args)
    slcuda_push_cuda(cuda_o);
}

// forward declaration, def below
static void _test_smooth (void);

static SLang_Intrin_Fun_Type Module_Intrinsics [] =
{
  MAKE_INTRINSIC_0("cusmooth", slcuda_smooth, SLANG_VOID_TYPE),
  MAKE_INTRINSIC_0("testsmooth", _test_smooth, SLANG_VOID_TYPE),
  SLANG_END_INTRIN_FUN_TABLE
};

  
int init_cudaimage_module_ns (char *ns_name)
{
   SLang_NameSpace_Type *ns;

   if (NULL == (ns = SLns_create_namespace (ns_name)))
     return -1;

   if ((-1 == SLns_add_intrin_fun_table (ns, Module_Intrinsics, NULL))
       )
     return -1;
   
   return 0;
}

void deinit_cudaimage_module (void)
{
}

// FOR TEST PURPOSES!
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
