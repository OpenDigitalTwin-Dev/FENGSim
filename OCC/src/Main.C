// file:   Main.C
// author: Jiping Xin

#include "TopoDS_Shape.hxx"
#include "BRepPrimAPI_MakeBox.hxx"

#include <iostream>

int main (int argv, char** argc) {
	std::cout << "test " << std::endl;

	TopoDS_Shape* S = new TopoDS_Shape(BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(1,0,0)),1,2,3).Shape());
    return 0;
}
