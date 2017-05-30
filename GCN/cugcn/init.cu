#include "luaT.h"
#include "THC.h"
#include "utils.h"

#include "GaborOrientationFilter.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcugcn(lua_State *L);

int luaopen_libcugcn(lua_State *L)
{
  lua_newtable(L);
  cugcn_GOF_init(L);
  return 1;
}
