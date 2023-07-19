#include <stdlib.h>
#include <iostream>
#include "Gmsh.h"
#include "GModel.h"

int main(int argc, char** argv) {
    GmshInitialize();
    GmshSetOption("Mesh","Algorithm",2.);
    //GmshSetOption("Mesh","MinimumCurvePoints",20.);
    GmshSetOption("Mesh","CharacteristicLengthMax", 1.0);
    GmshSetOption("General","Verbosity", 100.);

    // Create Gmsh model and set the factory
    GModel *GM = new GModel;
    GM->setFactory("OpenCASCADE");

    return 0;
}
