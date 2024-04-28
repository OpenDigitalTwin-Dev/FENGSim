// file: Solution.C
// author: Jiping Xin

#include "PoissonProblem.h"

void PoissonProblems::SetDomain (Mesh& M) {
    int id;
    ReadConfig(Settings, "EXAMPLE", id);
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (id) {
	case 1 : // thin wall
	    break;
	case 2 : // poisson mixed 2d
	    it->second->SetSubdomain(0);
	    break;
	case 3 : // poisson mixed 3d
	    it->second->SetSubdomain(0);
	    break;
	}
    }
}

void PoissonProblems::SetBoundary (Mesh& M) {
    int id;
    ReadConfig(Settings, "EXAMPLE", id);
    for (bnd_face bf = M.bnd_faces(); bf != M.bnd_faces_end(); bf++) {
	Point p(bf());
	hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);
	switch (id) {
	case 1 : {
	    if (pp->IsSource(bf)) {
		it->second.SetPart(1);
	    }
	    else if (bf()[2] == 0)
		it->second.SetPart(2);
	    else
		it->second.SetPart(3);
	    break;
	}
	case 2 :
	    if (p[0] == 0 || p[1] == 0)
		it->second.SetPart(111); 
	    else
		it->second.SetPart(222);
	    break;
	case 3 :
	    if (p[0] == 0 || p[1] == 0 || p[2] == 0)
		it->second.SetPart(1); 
	    else
		it->second.SetPart(0);
	    break;
	}
    }
}

bool PoissonProblems::IsDirichlet (int id) {
    switch (example_id) {
    case 1 :
	if (id == 1 || id == 2)
	    return true;
	return false;
	break;
    case 2 :
	if (id == 111)
	    return true;
	return false;
	break;
    case 3 : 
	return 1.0;
	break;
    }
}

double PoissonProblems::Coefficient (Point p) const { 
    switch (example_id) {
    case 1 : 
	return 0.1;
	break;
    case 2 : 
	return 1.0;
	break;
    case 3 : 
	return 1.0;
	break;
    }
}

double PoissonProblems::Source (Point p) const { 
    switch (example_id) {
    case 1 : 
	return 0;
	break;
    case 2 : 
	return -2.0 * exp(p[0] + p[1]);
	break;
    case 3 : 
	return -3.0 * exp(p[0] + p[1] + p[2]);
	break;
    }
    return 0.0;
}

double PoissonProblems::Dirichlet (Point p, int id) const { 
    switch (example_id) {
    case 1 :
	if (id == 1)
	    return 1;
	else if (id == 2)
	    return 0;
	break;
    case 2 :
	if (id == 111)
	    return exp(p[0] + p[1]); 
	break;
    case 3 : 
	return exp(p[0] + p[1] + p[2]); 
	break;
    }
}

double PoissonProblems::Neumann (Point p, int id) const {
    switch (example_id) {
    case 1 :
	if (id == 3)
	    return 0;
	break;
    case 2 :
	if (id == 222)
	    return exp(p[0] + p[1]); 
	break;
    case 3 :
	if (p[0] == 1)
	    return exp(p[0] + p[1] + p[2]);
	else if (p[1] == 1)
	    return exp(p[0] + p[1] + p[2]);
	else if (p[2] == 1)
	    return exp(p[0] + p[1] + p[2]); 
	break;
    }
}

double PoissonProblems::Solution (const Point& p) const { 
    switch (example_id) {
    case 1 : 
	return 0;
	break;
    case 2 : 
	return exp(p[0] + p[1]);
	break;
    case 3 : 
	return exp(p[0] + p[1] + p[2]);
	break;
    }
}
