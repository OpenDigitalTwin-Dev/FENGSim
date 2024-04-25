// file: Solution.C
// author: Jiping Xin

#include "AMProblem.h"

void AMProblem::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case EX1 : // heat dirichlet 2d
	    it->second->SetSubdomain(0);
	    break;
	case EX2 : // heat mixed 2d
	    it->second->SetSubdomain(0);
	    break;
	case EX3 : // heat mixed 3d
	    it->second->SetSubdomain(0);
	    break;
	case EX4 : // heat dirichlet 2d
	    it->second->SetSubdomain(0);
	    break;
	case EX5 : // heat from left to right
	    it->second->SetSubdomain(0);
	    break;
	}
    }
}

void AMProblem::SetBoundaryType (Mesh& M) {
    for (bnd_face bf = M.bnd_faces(); bf != M.bnd_faces_end(); bf++) {
	Point p(bf());
	hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);
	switch (example_id) {
	case EX1 :
	    it->second.SetPart(1); 
	    break;
	case EX2 :
	    if (p[0] == 1.0 || p[1] == 1.0) {
		it->second.SetPart(2);
		continue;
	    }
	    it->second.SetPart(1);
	    break;
	case EX3 :
	    if (p[0] == 0 || p[1] == 0 || p[2] == 0) {
		it->second.SetPart(1);
		continue;
	    }
	    it->second.SetPart(2);
	    break;
	case EX4 :
	    if (p[2] == 0) {
		it->second.SetPart(1);
		continue;
	    }
	    it->second.SetPart(2);
	    break;
	case EX5 :
	    if (p[2] == 0) {
		it->second.SetPart(1);
		continue;
	    }
	    it->second.SetPart(2);
	    break;
	}
    }
}

bool AMProblem::IsDirichlet (int id) {
    switch (example_id) {
    case EX1 :
        if (id == 1)
	    return true;
	return false;
	break;
    case EX2 :
	if (id == 1)
	    return true;
	return false;
	break;
    case EX3 :
        if (id == 1)
	    return true;
	return false;
	break;
    case EX4 :
        if (id == 1)
	    return true;
	return false;
	break;
    case EX5 :
	if (id == 1)
	    return true;
	return false;
	break;
    }
}

double AMProblem::SourceValue (Point p, double t) const { 
    switch (example_id) {
    case EX1 : 
        return -1.0 * exp(t) * exp(p[0] + p[1]);
	break;
    case EX2 : 
        return -1.0 * exp(t) * exp(p[0] + p[1]);
	break;
    case EX3 : 
	return -2.0 * exp(t) * exp(p[0] + p[1] + p[2]);
	break;
    case EX4 : {
        Point z = PP.CurrentPosition(t);
	if (PP.IsSource(z, p, 0.05, 0.05, 0.05)) {
	    return 0.1;
	}
	return 0;
	break;
    }
    case EX5 : {
        Point z = PP.CurrentPosition(t);
	if (PP.IsSource(z, p, 0.1, 0.1, 0.1)) {
	    return 0.01;
	}
	return 0;
	break;
    }
    }
    return 0.0;
}

double AMProblem::DirichletValue (Point p, int id, double t) const { 
    switch (example_id) {
    case EX1 :
        if (id == 1)
	    return exp(t) * exp(p[0] + p[1]); 
	break;
    case EX2 :
	if (id == 1)
	    return exp(t) * exp(p[0] + p[1]); 
	break;
    case EX3 :
	if (id == 1)
	    return exp(t) * exp(p[0] + p[1] + p[2]); 
	break;
    case EX4 :
        if (id == 1)
	    return 0; 
	break;
    case EX5 :
	if (id == 1)
	    return 0; 
	break;
    }
}

double AMProblem::NeumannValue (Point p, int id, double t) const {
    switch (example_id) {
    case EX1 :
	break;
    case EX2 :
	if (id == 2)
	    return exp(t) * exp(p[0] + p[1]); 
	break;
    case EX3 :
	if (id == 2)
	    return exp(t) * exp(p[0] + p[1] + p[2]);
    case EX4 :
      	if (id == 2)
	    return 0; 
	break;
    case EX5 :
	if (id == 2)
	    return 0; 
	break;
    }
}

double AMProblem::InitialValue (const Point& p) const { 
    switch (example_id) {
    case EX1 : 
	return exp(p[0] + p[1]);
	break;
    case EX2 : 
	return exp(p[0] + p[1]);
	break;
    case EX3 : 
	return exp(p[0] + p[1] + p[2]);
	break;
    case EX4 : 
	return 0;
	break;
    case EX5 :
	return 0;
	break;
    }
}

double AMProblem::Solution (const Point& p, double t) const { 
    switch (example_id) {
    case EX1 : 
        return exp(t) * exp(p[0] + p[1]);
	break;
    case EX2 :
        return exp(t) * exp(p[0] + p[1]);
	break;
    case EX3 : 
	return exp(t) * exp(p[0] + p[1] + p[2]);
	break;
    case EX4 : 
	return 0;
    case EX5 : 
	return 0;
	break;
    }
}
