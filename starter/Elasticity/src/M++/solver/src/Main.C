// file:   Main.C
// author: Christian Wieners Jiping Xin

#include "m++.h" 
#include "mpi.h"
#include "Elasticity.h"

#ifdef DCOMPLEX
#error undef DCOMPLEX in src/Compiler.h
#endif

void ElasticityMain ();
void TElasticityMain ();
void TElastoPlasticity2Main ();

int main (int argv, char** argc) {
	DPO dpo(&argv,&argc);
	string Model = "test";
	ReadConfig(Settings,"Model",Model);
	if (Model == "Elasticity") ElasticityMain();
	if (Model == "TElasticity") TElasticityMain();
	if (Model == "TElastoPlasticity2") TElastoPlasticity2Main();

    return 0;
}
