// file: Solution.C
// author: Jiping Xin

#include "ThermalElastoPlasticityProblems.h"

void ThermalElastoPlasticityProblems::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case 2 : // heat mixed 2d
	    it->second->SetSubdomain(0);
	    break;
	case 16 : // heat mixed 2d
	    it->second->SetSubdomain(0);
	    break;
	case 17 : // heat mixed 3d
	    it->second->SetSubdomain(0);
	    break;
	case 18 : // thinwall
	    it->second->SetSubdomain(0);
	    break;
	}
    }
}

void ThermalElastoPlasticityProblems::SetBoundaryType (Mesh& M) {
    for (bnd_face bf=M.bnd_faces(); bf!=M.bnd_faces_end(); bf++) {
	Point p(bf());
	hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);
	switch (example_id) {
	case 2 : {
	    if (p[0] == 0.0 || p[1] == 0.0) {
		it->second.SetPart(1); 
		continue;
	    }
	    else if (p[0] == 1.0) {
		it->second.SetPart(2); 
		continue;
	    }
	    else if (p[1] == 1.0) {
		it->second.SetPart(3); 
		continue;
	    }
	    break;
	}
	case 16 : {
	    if (p[0] == 0.0) {
		it->second.SetPart(1); 
		continue;
	    }
	    else if (p[1] == 0.0) {
		it->second.SetPart(2); 
		continue;
	    }
	    else if (p[0] == 1.0) {
		it->second.SetPart(3); 
		continue;
	    }
	    else if (p[1] == 1.0) {
		it->second.SetPart(4); 
		continue;
	    }
	    break;
	}
	case 17 : {
	    if (p[0] == 0.0) {
		it->second.SetPart(1); 
		continue;
	    }
	    else if (p[0] == 1.0) {
		it->second.SetPart(2); 
		continue;
	    }
	    else if (p[1] == 0.0) {
		it->second.SetPart(3); 
		continue;
	    }
	    else if (p[1] == 1.0) {
		it->second.SetPart(4); 
		continue;
	    }
	    else if (p[2] == 0.0) {
		it->second.SetPart(5); 
		continue;
	    }
	    else if (p[2] == 1.0) {
		it->second.SetPart(6); 
		continue;
	    }
	    break;
	}
	case 18 : {
	    if (p[2]==0.0) {
		it->second.SetPart(1); 
		continue;
	    }
	    it->second.SetPart(2);
	    break;
	}
	}
    }
}

bool ThermalElastoPlasticityProblems_T::IsDirichlet (int id) {
    switch (example_id) {
    case 2 : {
	if (id==1)
	    return true;
	return false;
	break;
    }
    case 16 : {
	if (id==2||id==4)
	    return true;
	return false;
	break;
    }
    case 17 : {
	if (id==5||id==6)
	    return true;
	return false;
	break;
    }
    case 18 : {
	if (id==1)
	    return true;
	return false;
	break;
    }
    }
}

double ThermalElastoPlasticityProblems_T::a (int id, Point p, double time) {
    switch (example_id) {
    case 2 : {
	return 1;
        return exp(time);
	break;
    }
    case 16 : {
        return 1.0;
	break;
    }
    case 17 : {
        return 1.0;
	break;
    }
    case 18 : {
        return 1.0;
	break;
    }
    }
}

double ThermalElastoPlasticityProblems_T::f (int id, Point p, double time) const { 
    switch (example_id) {
    case 2 : {
	return exp(time) * exp(p[0] + p[1]) - 2.0 * exp(time) * exp(p[0] + p[1]);
        return exp(time) * exp(p[0] + p[1]) - exp(time) * 2.0 * exp(time) * exp(p[0] + p[1]);
	break;
    }
    case 16 : {
        return 0;
	break;
    }
    case 17 : {
        return 0;
	break;
    }
    case 18 : {
	if (p[2]>time*10+2&&p[2]<time*10+0.25+2)
	    return 10;
	else
	    return 0;
	break;
    }
    }
    return 0.0;
}

void ThermalElastoPlasticityProblems_T::g_D (int k, RowBndValues& u_c, int id, Point p, double time) { 
    switch (example_id) {
    case 2 : {
	u_c.D(k,0) = true;
	u_c(k,0) = exp(time) * exp(p[0] + p[1]);
	break;
    }
    case 16 : {
	if (id == 2) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0.1;
	    break;
	}
	if (id == 4) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0;
	    break;
	}
    }
    case 17 : {
	if (id == 5) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0.1;
	    break;
	}
	if (id == 6) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0;
	    break;
	}
    }
    case 18 : {
	if (id == 1) {
	    u_c.D(k,0) = true;
	    u_c(k,0) = 0.0;
	    break;
	}
    }
    }
}

double ThermalElastoPlasticityProblems_T::g_N (int id, Point p, double time) const {
    switch (example_id) {
    case 2 : {
	return exp(time) * exp(p[0] + p[1]);
	return exp(time) * exp(time) * exp(p[0] + p[1]);
	break;
    }
    case 16 : {
      	if (id == 1 || id == 3)
	    return 0;
	break;
    }
    case 17 : {
      	if (id == 1 || id == 2 || id == 3 || id == 4)
	    return 0;
	break;
    }
    case 18 : {
      	if (id == 2)
	    return -0.01;
	break;
    }
    }
}

double ThermalElastoPlasticityProblems_T::u0 (const Point& p) const { 
    switch (example_id) {
    case 2 :
	return exp(p[0] + p[1]);
	break;
    case 16 :
	return 0;
	break;
    case 17 :
	return 0;
	break;
    case 18 :
	return 0;
	break;
    }
}

double ThermalElastoPlasticityProblems_T::u (const Point& p, double time) const { 
    switch (example_id) {
    case 2 : {
	return exp(time) * exp(p[0] + p[1]);
	break;
    }
    case 16 :
        return 0;
	break;
    case 17 :
        return 0;
	break;
    case 18 :
        return 0;
	break;
    }
}

bool ThermalElastoPlasticityProblems_D::IsDirichlet (int id) {
    switch (example_id) {
    case 2 : {
	if (id == 111) {
	    return true;
	}
	return false;
	break;
    }
    }
}

void ThermalElastoPlasticityProblems_D::DirichletValue (int k, RowBndValues& u_c, int id, Point p, double time) {
    switch (example_id) {
    case 2 : {
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

Point ThermalElastoPlasticityProblems_D::NeumannValue (int id, Point p, double t) {
    switch (example_id) {
    case 2 :
      	if (id == 222) {
	    return zero;
	}
	break;
    }
}

Point ThermalElastoPlasticityProblems_D::SourceValue (int id, Point p, double t) {
    switch (example_id) {
    case 2 : {
	if (id == 0) {
	    return zero;
	}
	break;
    }
    }
    return zero;
}

Point ThermalElastoPlasticityProblems_D::Solution (Point p, double t) {
    Point pp = zero;
    switch (example_id) {
    case 2 :
        return zero;
	break;
    }
}
