// file:   Main.C
// author: Jiping Xin

#include "m++.h" 
#include "mpi.h"

#ifdef DCOMPLEX
#error undef DCOMPLEX in src/Compiler.h
#endif

void SlicePhaseTestMain (int argc, char** argv);
void InfillTestMain ();
void AMMain ();
void PoissonMain ();
void HeatMain ();
void ElasticityMain ();
void TElasticityMain ();
void TElastoPlasticity2Main ();
void inp2geo ();

int main (int argc, char** argv) {
    DPO dpo(&argc,&argv);
    string Model = "test";
    ReadConfig(Settings,"Model",Model);
    if (Model == "SlicePhaseTest") SlicePhaseTestMain(argc, argv);
    if (Model == "InfillTest") InfillTestMain();
    if (Model == "AM") AMMain();
    if (Model == "Poisson") PoissonMain();
    if (Model == "Heat") HeatMain();
    if (Model == "Elasticity") ElasticityMain();
    if (Model == "TElasticity") TElasticityMain();
    if (Model == "TElastoPlasticity2") TElastoPlasticity2Main();
    if (Model == "inp2geo") inp2geo();
    return 0;
}
