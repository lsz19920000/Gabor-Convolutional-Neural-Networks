template <typename Dtype>
__global__ void GaborProducingKernel(
    const int nthreads, 
    const Dtype* weight_data,
    const Dtype* gaborFilterBank_data,
    const int nInputPlane,
    const int nOutputPlane,
    const int nChannel,
    const int nEntry,
    Dtype* output_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        long l = n % nEntry;
        long j = (n / nEntry) % nInputPlane;
        long i = n / nEntry / nInputPlane;
        long k;
        float val = *(weight_data + n);
        for (k = 0; k < nChannel; k++) {
	    float gabortmp=*(gaborFilterBank_data+k*(nEntry / nChannel)+l%(nEntry / nChannel));
            float *target = output_data + i * (nChannel * nInputPlane * nEntry)
                                        + k * (nInputPlane * nEntry)
                                        + j * (nEntry)
                                        + l;
            *target = val*gabortmp;
        }
    }
}

static int cugcn_GOF_Producing(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gaborFilterBank = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    long kW = lua_tonumber(L, 5);
    long kH = lua_tonumber(L, 6);
    long nInputPlane = lua_tonumber(L, 7);
    long nOutputPlane = lua_tonumber(L, 8);
    long nChannel = lua_tonumber(L, 9);

    THCUNN_assertSameGPU(state, 3, weight, gaborFilterBank, output);

    float *weight_data = THCudaTensor_data(state, weight);
    float *gaborFilterBank_data = THCudaTensor_data(state, gaborFilterBank);
    float *output_data = THCudaTensor_data(state, output);

    int nEntry = nChannel * kH * kW;
    long count = nOutputPlane * nInputPlane * nEntry;

    GaborProducingKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, weight_data, gaborFilterBank_data, nInputPlane, nOutputPlane, nChannel, nEntry, output_data);
    THCudaCheck(cudaGetLastError());

    return 1;
}

template <typename Dtype>
__global__ void BPAlignKernel(
    const int nthreads, 
    const Dtype* gradWeight_data,
    const int nInputPlane,
    const int nOutputPlane,
    const int nChannel,
    const int kH,
    const int kW,
    Dtype* weight_data,
    const Dtype* gaborFilterBank_data) 
{
    int nEntry=nChannel*kH*kW;
    CUDA_KERNEL_LOOP(n, nthreads) {
        long l = n % nEntry;
        long j = (n / nEntry) % nInputPlane;
        long i = n / nEntry / nInputPlane;
        long k;
        float *val = weight_data + n;
        for (k = 0; k < nChannel; k++) {
            float gabortmp=*(gaborFilterBank_data+k*(kW*kH)+l%(kW*kH));
            float target = *(gradWeight_data + i * (nChannel * nInputPlane * nEntry)
                                             + k * (nInputPlane * nEntry)
                                             + j * (nEntry)
                                             + l);
            
			*val = *val + target*gabortmp;
			//*val = *val + target;
        }
    }
}

static int cugcn_GOF_BPAlign(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *gradWeight = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    long kW = lua_tonumber(L, 4);
    long kH = lua_tonumber(L, 5);
    long nInputPlane = lua_tonumber(L, 6);
    long nOutputPlane = lua_tonumber(L, 7);
    long nChannel = lua_tonumber(L, 8);
    THCudaTensor *gaborFilterBank = (THCudaTensor*)luaT_checkudata(L, 11, "torch.CudaTensor");
	
    THCUNN_assertSameGPU(state, 3, weight, gradWeight, gaborFilterBank);

    float *weight_data = THCudaTensor_data(state, weight);
    float *gaborFilterBank_data = THCudaTensor_data(state, gaborFilterBank);
    float *gradWeight_data = THCudaTensor_data(state, gradWeight);

    int nEntry = nChannel * kH * kW;
    long count = nOutputPlane * nInputPlane * nEntry;

    BPAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count, gradWeight_data, nInputPlane, nOutputPlane, nChannel, kH, kW, weight_data, gaborFilterBank_data);
    THCudaCheck(cudaGetLastError());

    return 1;
}

static const struct luaL_Reg cugcn_GOF__ [] = {
    {"GOF_Producing", cugcn_GOF_Producing},
    {"GOF_BPAlign", cugcn_GOF_BPAlign},
    {NULL, NULL}
};

static void cugcn_GOF_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cugcn_GOF__, "gcn");
    lua_pop(L,1);
}
