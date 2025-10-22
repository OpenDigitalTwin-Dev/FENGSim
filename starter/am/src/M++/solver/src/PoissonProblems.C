// file: Solution.C
// author: Jiping Xin

#include "PoissonProblems.h"

void PoissonProblems::SetDomain (Mesh& M) {
    int id;
    ReadConfig(Settings, "EXAMPLE", id);
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (id) {
	case 1 : // poisson dirichlet 2d
	    it->second->SetSubdomain(0);
	    break;
	case 2 : // poisson mixed 2d
	    it->second->SetSubdomain(0);
	    break;
	case 3 : // poisson mixed 3d
	    it->second->SetSubdomain(0);
	    break;
	case 4 : // domain decomposition method
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
	case 1 :
	    it->second.SetPart(1); 
	    break;
	case 2 :
	    if (p[0] == 0 || p[1] == 0)
		it->second.SetPart(1); 
	    else
		it->second.SetPart(2);
	    break;
	case 3 :
	    if (p[0] == 0 || p[1] == 0 || p[2] == 0)
		it->second.SetPart(1); 
	    else
		it->second.SetPart(2);
	    break;
	case 4 :
	    if (p[0] == 0 || p[1] == 0 || p[0] == 3 || p[1] == 3)
		it->second.SetPart(1); 
	    else
		it->second.SetPart(2);
	    break;
	}
    }
}

bool PoissonProblems::IsDirichlet (int id) {
    switch (example_id) {
    case 1 : 
	return 1.0;
	break;
    case 2 :
	if (id == 1)
	    return true;
	return false;
	break;
    case 3 :
        if (id == 1)
	    return true;
	return false;
	break;
    case 4 :
        if (id == 1)
	    return true;
	return false;
	break;
    }
}

double PoissonProblems::alpha (Point p) const { 
    switch (example_id) {
    case 1 : 
	return 1.0;
	break;
    case 2 : 
	return 1.0;
	break;
    case 3 : 
	return 1.0;
	break;
    case 4 : 
	return 1.0;
	break;
    }
}

double PoissonProblems::f (Point p) const { 
    switch (example_id) {
    case 1 : 
	return -2.0 * exp(p[0] + p[1]);
	break;
    case 2 : 
	return -2.0 * exp(p[0] + p[1]);
	break;
    case 3 : 
	return -3.0 * exp(p[0] + p[1] + p[2]);
	break;
    case 4 : 
	return -2.0 * exp(p[0] + p[1]);
	break;
    }
    return 0.0;
}

double PoissonProblems::g_D (Point p, int id) const { 
    switch (example_id) {
    case 1 :
	return exp(p[0] + p[1]); 
	break;
    case 2 :
	if (id == 1)
	    return exp(p[0] + p[1]); 
	break;
    case 3 :
	if (id == 1)
	    return exp(p[0] + p[1] + p[2]); 
	break;
    case 4 :
	if (id == 1)
	    return exp(p[0] + p[1]); 
	break;
    }
}

double PoissonProblems::g_N (Point p, int id) const {
    switch (example_id) {
    case 1 :
	break;
    case 2 :
	if (id == 2)
	    return exp(p[0] + p[1]); 
		break;
    case 3 :
	if (id == 2)
	    return exp(p[0] + p[1] + p[2]); 
	break;
    }
}

double PoissonProblems::u (Point p) const { 
    switch (example_id) {
    case 1 : 
	return exp(p[0] + p[1]);
	break;
    case 2 : 
	return exp(p[0] + p[1]);
	break;
    case 3 : 
	return exp(p[0] + p[1] + p[2]);
	break;
    case 4 : 
	return exp(p[0] + p[1]);
	break;
    }
}
