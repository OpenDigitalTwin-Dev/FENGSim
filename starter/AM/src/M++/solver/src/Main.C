// file:   Main.C
// author: Christian Wieners Jiping Xin

#include "m++.h" 
#include "mpi.h"

#ifdef DCOMPLEX
#error undef DCOMPLEX in src/Compiler.h
#endif

void SlicePhaseTestMain (int argc, char** argv);
void InfillTestMain ();
void AMMain ();

int main (int argc, char** argv) {
    DPO dpo(&argc,&argv);
    string Model = "test";
    ReadConfig(Settings,"Model",Model);
    if (Model == "SlicePhaseTest") SlicePhaseTestMain(argc, argv);
    if (Model == "InfillTest") InfillTestMain();
    if (Model == "AM") AMMain();
    return 0;
}
