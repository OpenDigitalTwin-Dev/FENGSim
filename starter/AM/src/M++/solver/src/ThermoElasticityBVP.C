// file: Solution.C
// author: Jiping Xin

#include "ThermoElasticityBVP.h"

void ThermoElasticityGeoID::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case 1 : // thermoelasticity mixed 2d abaqus
	  //it->second->SetSubdomain(0);
	    break;
	}
    }
}

void ThermoElasticityGeoID::SetBoundaryType (Mesh& M) {
    for (bnd_face bf = M.bnd_faces(); bf != M.bnd_faces_end(); bf++) {
	Point p(bf());
	hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);
	switch (example_id) {
	case 1 :
	    if (p[2] == 0.0) {
		it->second.SetPart(111);
		continue;
	    }
	    it->second.SetPart(222);
	    break;
	}
    }
}

bool ThermoElasticityBVP_T::IsDirichlet (int id) {
    switch (example_id) {
    case 1 :
	if (id == 111)
	    return true;
	return false;
	break;
    }
}

double ThermoElasticityBVP_T::DiffusionCoe (int id, Point p, double t) {
    switch (example_id) {
    case 4 : 
        return 1.0;
	break;
    case 5 : 
        return 1;
	break;
    case 6 :
	return 0.05;
	break;
    }
}

double ThermoElasticityBVP_T::SourceValue (int id, Point p, double t) const { 
    switch (example_id) {
    case 4 : 
        return 0;
	break;
    case 5 : {
        return 0;
	break;
    }
    case 6 :
	double v = 1.0;
	if (p[0] > t*v-0.01 && p[0] < t*v+0.01)
	    if (p[2] < 1 && p[2] > 0.95)
	        return 50 * exp(-(p[0] - t*v) * (p[0] - t*v) * 10.0);
	return 0;
	break;
    }
    return 0.0;
}

double ThermoElasticityBVP_T::DirichletValue (int id, Point p, double t) const { 
    switch (example_id) {
    case 4 :
        if (id == 444)
	    return 1.0;
	break;
    case 5 : {
	if (id == 4)
	    return 1.0;
	break;
    }
    case 6 :
        if (id == 333)
	    return 0;
	break;
    }
}

double ThermoElasticityBVP_T::NeumannValue (int id, Point p, double t) const {
    switch (example_id) {
    case 4 : {
      	if (id == 111 || id == 222 || id == 333)
	    return 0;
	break;
    }
    case 5 : {
        if (id==1||id==2||id==3||id==5||id==6)
	    return 0;
	break;
    }
    case 6 :
      	if (id == 111 || id == 222 || id == 444 || id == 555 || id == 666)
	    return 0;
	break;
    }
}

double ThermoElasticityBVP_T::InitialValue (const Point& p) const { 
    switch (example_id) {
    case 4 :
	return 0;
	break;
    case 5 : {
        return 0;
	break;
    }
    case 6 :
	return 0;
	break;
    }
}

double ThermoElasticityBVP_T::Solution (const Point& p, double t) const { 
    switch (example_id) {
    case 4 :
        return 0;
	break;
    case 5 : {
        return 0;
	break;
    }
    case 6 :
        return 0;
	break;
    }
}

bool ThermoElasticityBVP_D::IsDirichlet (int id) {
    switch (example_id) {
    case 1 : {
	if (id == 111) {
	    return true;
	}
	return false;
	break;
    }
    }
}

void ThermoElasticityBVP_D::DirichletValue (int k, RowBndValues& u_c, int id, Point p) {
    switch (example_id) {
    case 1 : {
        if (id == 111) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0;
	    u_c.D(k,1) = true;
	    u_c(k,1) = 0;
	    u_c.D(k,2) = true;
	    u_c(k,2) = 0;
	}
	break;
    }
    }
}

Point ThermoElasticityBVP_D::NeumannValue (int id, Point p, double t) {
    switch (example_id) {
    case 1 :
      	if (id == 222) {
	    return zero;
	}
	break;
    }
}

Point ThermoElasticityBVP_D::SourceValue (int id, Point p, double t) {
    switch (example_id) {
    case 1 : {
	if (id == 0) {
	    return zero;
	}
	break;
    }
    }
    return zero;
}

Point ThermoElasticityBVP_D::Solution (Point p, double t) {
    Point pp = zero;
    switch (example_id) {
    case 1 :
        return zero;
	break;
    }
}
