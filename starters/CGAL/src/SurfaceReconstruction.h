#ifndef _SURFACERECONSTRUCTION_H_
#define _SURFACERECONSTRUCTION_H_


using namespace std;

#include <stdio.h>
#include <vector>


void AlphaShape2 (std::vector<double> pc, double a=0, int mesher=-1, double p1=0.125, double p2=0.1);
void CGALMeshGeneration();


#endif
