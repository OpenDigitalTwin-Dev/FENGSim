// file:   Main.C
// author: Christian Wieners Jiping Xin

#include "m++.h" 
#include "mpi.h"

#ifdef DCOMPLEX
#error undef DCOMPLEX in src/Compiler.h
#endif

void AMMain ();

int main (int argv, char** argc) {
    DPO dpo(&argv,&argc);
    string Model = "test";
    ReadConfig(Settings,"Model",Model);
    if (Model == "AM") AMMain();
    return 0;
}
