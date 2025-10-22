// file: Solution.C
// author: Jiping Xin

#include "HeatProblems.h"

void HeatProblems::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
		Point p(c());
		hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
		switch (example_id) {
		case 1 : // heat dirichlet 2d
			it->second->SetSubdomain(0);
			break;
		case 2 : // heat mixed 2d
			it->second->SetSubdomain(0);
			break;
		case 3 : // heat mixed 3d
			it->second->SetSubdomain(0);
			break;
		}
    }
}

void HeatProblems::SetBoundaryType (Mesh& M) {
    for (bnd_face bf = M.bnd_faces(); bf != M.bnd_faces_end(); bf++) {
		Point p(bf());
		hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);
		switch (example_id) {
		case 1 :
			it->second.SetPart(1); 
			break;
		case 2 :
			if (p[0] == 0.0 || p[1] == 0.0) {
				it->second.SetPart(1);
				continue;
			}
			it->second.SetPart(2);
			break;
		case 3 :
			if (p[0] == 0 || p[1] == 0 || p[2] == 0) {
				it->second.SetPart(1);
				continue;
			}
			it->second.SetPart(2);
			break;
		}
    }
}

bool HeatProblems::IsDirichlet (int id) {
    switch (example_id) {
    case 1 :
        if (id == 1)
			return true;
		return false;
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
    }
}

double HeatProblems::f (Point p, double t) const { 
    switch (example_id) {
    case 1 : 
        return -1.0 * exp(t) * exp(p[0] + p[1]);
		break;
    case 2 : 
        return -1.0 * exp(t) * exp(p[0] + p[1]);
		break;
    case 3 : 
		return -2.0 * exp(t) * exp(p[0] + p[1] + p[2]);
		break;
    }
    return 0.0;
}

double HeatProblems::g_D (Point p, int id, double t) const { 
    switch (example_id) {
    case 1 :
        if (id == 1)
			return exp(t) * exp(p[0] + p[1]); 
		break;
    case 2 :
		if (id == 1)
			return exp(t) * exp(p[0] + p[1]); 
		break;
    case 3 :
		if (id == 1)
			return exp(t) * exp(p[0] + p[1] + p[2]); 
		break;
    }
}

double HeatProblems::g_N (Point p, int id, double t) const {
    switch (example_id) {
    case 1 :
		break;
    case 2 :
		if (id == 2)
			return exp(t) * exp(p[0] + p[1]); 
		break;
    case 3 :
		if (id == 2)
			return exp(t) * exp(p[0] + p[1] + p[2]);
    }
}

double HeatProblems::v (const Point& p) const { 
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
    }
}

double HeatProblems::u (const Point& p, double t) const { 
    switch (example_id) {
    case 1 : 
        return exp(t) * exp(p[0] + p[1]);
		break;
    case 2 :
        return exp(t) * exp(p[0] + p[1]);
		break;
    case 3 : 
		return exp(t) * exp(p[0] + p[1] + p[2]);
		break;
    }
}
