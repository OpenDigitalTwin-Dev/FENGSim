// file:   Main.C
// author: Christian Wieners Jiping Xin

#include "m++.h" 
#include "mpi.h"
#include "Elasticity.h"

#ifdef DCOMPLEX
#error undef DCOMPLEX in src/Compiler.h
#endif

void ElasticityMain ();
void ContactMain ();

int main (int argv, char** argc) {
	DPO dpo(&argv,&argc);
	string Model = "test";
	ReadConfig(Settings,"Model",Model);
	if (Model == "Elasticity") ElasticityMain();
	if (Model == "Contact") ContactMain();

    return 0;
}
