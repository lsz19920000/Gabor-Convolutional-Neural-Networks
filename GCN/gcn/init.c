#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define gcn_(NAME) TH_CONCAT_3(gcn_, Real, NAME)

#include "generic/GaborOrientationFilter.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libgcn(lua_State *L);

int luaopen_libgcn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "gcn");

  gcn_FloatGOF_init(L);
  gcn_DoubleGOF_init(L);

  return 1;
}
