// file: Main.C
// author: Christian Wieners
// $Header: /public/M++/src/Main.C,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#include "m++.h"

void ELaplace();
void EError();
void Plasticity();
void Maxwell();
void Stokes();
void LinearElasticityMain ();

int main (int argv, char** argc) {
    DPO dpo(&argv,&argc);
    string Model = "ELaplace";  ReadConfig(Settings,"Model",Model);
    if      (Model == "ELaplace")    ELaplace();
    else if (Model == "EError")      EError();
//    else if (Model == "Plasticity")  Plasticity();
    else if (Model == "Maxwell")     Maxwell();
//    else if (Model == "Stokes")      Stokes();
//    else                             TestProblem();
    else if (Model == "Linear")     LinearElasticityMain ();
    return 0;
}

