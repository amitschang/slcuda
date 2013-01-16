import("cuda");
require("rand");

%
% these should come from the makefile variables
%
variable NVCC_BIN="/usr/local/cuda/bin/nvcc";
variable NVCC_FLAGS="-shared";
variable NVCC_C_FLAGS="-fPIC";
variable NVCC_INC="-I/usr/local/cuda/include";
variable NVCC_LIB="-L/usr/local/cuda/lib64";

variable CURAND_DEFAULT_GEN_TYPE=CURAND_RNG_PSEUDO_DEFAULT;
variable CURAND_DEFAULT_GEN;

private define nvcompileandimport(modname){
    variable r;
    variable odir=getcwd();
    variable opath=get_import_module_path();
    ()=chdir("/tmp");
    variable cmd=strjoin([NVCC_BIN,
                      NVCC_FLAGS,
                      "--compiler-options",
                      NVCC_C_FLAGS,
                      NVCC_INC,
                      NVCC_LIB,
                      "-lslang -lcudart",
                      modname+".cu",
                      "-o "+modname+"-module.so"],
                     " ");
    r=system(cmd+" >/dev/null 2>&1");
    set_import_module_path(".");
    try {
	import(modname);
	r=1;
    }
    catch ImportError: {
	fprintf(stderr,"Unable to import dynamic CUDA code at /tmp/%s.cu",modname);
	r=0;
    }
    set_import_module_path(opath);
    ()=chdir(odir);
    return r;
}

private define parse_kernel_def(kernel){
    variable name,i;
    variable args=DataType_Type[0];
    % 
    % compact spaces
    % 
    kernel=strcompress(kernel," \n");
    %
    % first get the name
    % 
    variable tmp=strchop(kernel,'(',0);
    tmp=strchop(tmp[0],' ',0);
    name=tmp[2];
    tmp=strchop(kernel,'(',0);
    tmp=strchop(tmp[1],')',0)[0];
    tmp=strchop(tmp,',',0);
    foreach i (tmp){
	i=strtrim(i);
	if (string_match(i,"*")){
	    args=[args,SLcuda_Type];
	}
	else if (string_match(i,"^int")){
	    args=[args,Integer_Type];
	}
	else {
	    args=[args,Double_Type];
	}
    }
    return (name,args);
}

% 
% Generate code from a kernel definition
%
define cuda_add_function(name, kernel){
    variable i,n=0;
    variable kname,input_types;
    (kname,input_types)=parse_kernel_def(kernel);
    if (input_types[-1]!=Integer_Type)
	verror("The last argument to kernel must be integer type denoting data length");
    else
        input_types=input_types[[:-2]];
    variable modname="slcuda_"+name+string(rand);
    variable fp=fopen("/tmp/"+modname+".cu","w");
    %
    % write header
    % 
    ()=fputs("#include <slang.h>\n#include <cuda.h>\n",fp);
    ()=fputs("SLANG_MODULE("+modname+");\n",fp);
    %
    % write the kernel definition
    % 
    ()=fputs(kernel+"\n",fp);
    % 
    % define slcuda_type, should probably include from external source
    % 
    ()=fprintf(fp,"typedef struct{int devid;int size;int ndims;");
    ()=fprintf(fp,"int nelems;int valid;SLindex_Type dims[3];");
    ()=fprintf(fp,"void *dptr;}SLcuda_Type;\n");
    % 
    % Now, write the handler function, using the input_types list to
    % parse input args
    % 
    ()=fputs("static void handler (void)\n{\n",fp);
    for (i=length(input_types)-1;i>=0;i--){
	if (input_types[i]==SLcuda_Type){
	    ()=fprintf(fp,"SLang_MMT_Type *mmt_%d;\n",i);
	    ()=fprintf(fp,"SLcuda_Type *cuda_%d;\n",i);
	    ()=fprintf(fp,"float *arg_%d;\n",i);
	    ()=fprintf(fp,"if(NULL==(mmt_%d=SLang_pop_mmt(%d))){ return; }\n",i,__class_id(SLcuda_Type));
	    ()=fprintf(fp,"if(NULL==(cuda_%d=(SLcuda_Type *)SLang_object_from_mmt(mmt_%d))){ return; }\n",i,i);
	    ()=fprintf(fp,"arg_%d=(float *)cuda_%d->dptr;\n",i,i);
	    n=i;
	}
	else if (input_types[i]==Integer_Type){
	    ()=fprintf(fp,"int arg_%d;\n",i);
	    ()=fprintf(fp,"if(-1==SLang_pop_int(&arg_%d)){ return; }\n",i);
	}
	else if (input_types[i]==Double_Type){
	    ()=fprintf(fp,"double arg_%d_in;\n",i);
	    ()=fprintf(fp,"float  arg_%d;\n",i);
	    ()=fprintf(fp,"if(-1==SLang_pop_double(&arg_%d_in)){ return; }\n",i);
	    ()=fprintf(fp,"arg_%d=(float)arg_%d_in;\n",i,i);
	}
    }
    ()=fprintf(fp,"int N=cuda_%d->nelems;\n",n);
    ()=fprintf(fp,"int block_size = %d;\n",CUDA_BLOCK_SIZE);
    ()=fprintf(fp,"int n_blocks = N/block_size;\n");
    ()=fprintf(fp,"if ((n_blocks*N)<N) n_blocks++;\n");
    ()=fputs(kname+" <<< n_blocks, block_size >>> (",fp);
    for (i=0;i<length(input_types);i++){
	if (i==0) ()=fprintf(fp,"arg_0");
	else  ()=fprintf(fp,",arg_%d",i);
    }
    ()=fputs(",N",fp);
    ()=fputs(");\n}\n",fp);
    ()=fputs("int init_"+modname+"_module_ns (char *ns_name){\n",fp);
    ()=fputs("SLang_NameSpace_Type *ns;\n",fp);
    ()=fputs("if (NULL == (ns = SLns_create_namespace (ns_name)))\n",fp);
    ()=fputs("return -1;\n",fp);
    ()=fprintf(fp,"SLadd_intrinsic_function(\"%s\",",name);
    ()=fprintf(fp,"(FVOID_STAR)handler,SLANG_VOID_TYPE,0);\n");
    ()=fputs("return 0;\n}",fp);
    ()=fclose(fp);
    if (nvcompileandimport(modname)){
	()=remove("/tmp/"+modname+".cu");
	()=remove("/tmp/"+modname+"-module.so");
    }
}

define cuda_add_function_file(filename){
    variable name;
    if (_NARGS==2){
	name=();
    }
    else {
	name=path_basename_sans_extname(filename);
    }
    variable fp=fopen(filename,"r");
    variable kernel=strjoin(fgetslines(fp),"\n");
    ()=fclose(fp);
    cuda_add_function(name,kernel);
}

define curand_get_default_gen (){
    if (not __is_initialized(&CURAND_DEFAULT_GEN)){
	CURAND_DEFAULT_GEN=curand_new(CURAND_DEFAULT_GEN_TYPE,[_time,getpid]);
    }
    return CURAND_DEFAULT_GEN;
}

define curand (){
    variable arg=();
    variable gen;
    variable cuda;

    if (_NARGS==2){
	gen=();
    }
    else {
	gen=curand_get_default_gen();
    }
    if (typeof(arg)==Integer_Type){
	cuda=cuarr(arg);
    }
    else {
	cuda=arg;
    }
    curand_gen(CURAND_DEFAULT, gen, cuda);
    if (typeof(arg)==Integer_Type){
	return cuda;
    }
}

define curand_uniform (){
    variable arg=();
    variable gen;
    variable cuda;

    if (_NARGS==2){
	gen=();
    }
    else {
	gen=curand_get_default_gen();
    }
    if (typeof(arg)==Integer_Type){
	cuda=cuarr(arg);
    }
    else {
	cuda=arg;
    }
    curand_gen(CURAND_UNIFORM, gen, cuda);
    if (typeof(arg)==Integer_Type){
	return cuda;
    }
}

define curand_normal (){
    variable arg=();
    variable sigma=();
    variable mean=();
    variable gen;
    variable cuda;

    if (_NARGS==4){
	gen=();
    }
    else {
	gen=curand_get_default_gen();
    }
    if (typeof(arg)==Integer_Type){
	cuda=cuarr(arg);
    }
    else {
	cuda=arg;
    }
    curand_gen(CURAND_NORMAL, gen, cuda, mean, sigma);
    if (typeof(arg)==Integer_Type){
	return cuda;
    }
}

define curand_lognormal(){
    variable arg=();
    variable sigma=();
    variable mean=();
    variable gen;
    variable cuda;

    if (_NARGS==4){
	gen=();
    }
    else {
	gen=curand_get_default_gen();
    }
    if (typeof(arg)==Integer_Type){
	cuda=cuarr(arg);
    }
    else {
	cuda=arg;
    }
    curand_gen(CURAND_LOGNORMAL, gen, cuda, mean, sigma);
    if (typeof(arg)==Integer_Type){
	return cuda;
    }
}
