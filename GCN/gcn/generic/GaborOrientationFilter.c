#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GaborOrientationFilter.c"
#else
static int gcn_(GOF_Producing)(lua_State *L)
{
    THTensor *weight = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *gaborFilterBank = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *output = luaT_checkudata(L, 4, torch_Tensor);
    long kW = lua_tonumber(L, 5);
    long kH = lua_tonumber(L, 6);
    long nInputPlane = lua_tonumber(L, 7);
    long nOutputPlane = lua_tonumber(L, 8);
    long nChannel = lua_tonumber(L, 9);


    real *weightData = THTensor_(data)(weight);
    real *gaborFilterBankData = THTensor_(data)(gaborFilterBank);
    real *outputData = THTensor_(data)(output);
	
    long i, j, l, k;
  

#pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            
            for (l = 0; l < nChannel * kH * kW; l++) {
                real val = *(weightData + i * (nInputPlane * nChannel * kH * kW)
                                        + j * (nChannel * kH * kW)
                                        + l);
										
                
                for (k = 0; k < nChannel; k++) {
		        real gabortmp=*(gaborFilterBankData+k*(kW*kH)+l%(kW*kH));

                    real *target = outputData + i * (nChannel * nInputPlane * nChannel * kH * kW)
                                              + k * (nInputPlane * nChannel * kH * kW)
                                              + j * (nChannel * kH * kW)
                                              + l;
                    *target = val*gabortmp;
                }
            }
        }
    }

    return 1;
}

static int gcn_(GOF_BPAlign)(lua_State *L)
{
    THTensor *weight = luaT_checkudata(L, 2, torch_Tensor);  
    THTensor *gradWeight = luaT_checkudata(L, 3, torch_Tensor);
    long kW = lua_tonumber(L, 4);
    long kH = lua_tonumber(L, 5);
    long nInputPlane = lua_tonumber(L, 6);
    long nOutputPlane = lua_tonumber(L, 7);
    long nChannel = lua_tonumber(L, 8);
    THTensor *gaborFilterBank = luaT_checkudata(L, 11, torch_Tensor);


    real *weightData = THTensor_(data)(weight);
    real *gaborFilterBankData = THTensor_(data)(gaborFilterBank);
    real *gradWeightData = THTensor_(data)(gradWeight);

    long nEntry = nChannel * kH * kW;
    long i, j, l, k;


#pragma omp parallel for private(i)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            for (l = 0; l < nEntry; l++) {
                real *val = weightData + i * (nInputPlane * nEntry)
                                       + j * (nEntry)
                                       + l;
                for (k = 0; k < nChannel; k++) {

                    real gabortmp=*(gaborFilterBankData+k*(kW*kH)+l%(kW*kH));
                    real *target = gradWeightData + i * (nChannel * nInputPlane * nEntry)
                                                  + k * (nInputPlane * nEntry)
                                                  + j * (nEntry)
                                                  + l;
                   *val = *val + *target*gabortmp;
					 /**val = *val + *target;*/
                }
            }
        }
    }

    return 1;
}

static const struct luaL_Reg gcn_(GOF__) [] = {
    {"GOF_Producing", gcn_(GOF_Producing)},
    {"GOF_BPAlign", gcn_(GOF_BPAlign)},
    {NULL, NULL}
};

static void gcn_(GOF_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, gcn_(GOF__), "gcn");
    lua_pop(L,1);
}

#endif
