// file: Solution.C
// author: Jiping Xin

#include "TElasticityProblems.h"

void TElasticityProblems::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case 0 : // fengsim 
	    break;
	case 1 : // 2d dirichlet boundary condition
	    it->second->SetSubdomain(0);
	    break;
	case 2 : // 2d mixed boundary condition 
	    it->second->SetSubdomain(0);
	    break;
	case 3 : // 3d elasticity dirichlet b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 4 : // 3d elasticity mixed b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 5 : // traction
	    it->second->SetSubdomain(0);
	    break;
	case 6 : // impaction
	    it->second->SetSubdomain(0);
	    break;
	case 7 : // rotation
	    it->second->SetSubdomain(0);
	    break;
	}
    }
}

void TElasticityProblems::SetBoundaryType (Mesh& M) {
    for (bnd_face bf = M.bnd_faces(); bf != M.bnd_faces_end(); bf++) {
	Point p(bf());
	hash_map<Point,BoundaryFace,Hash>::iterator it = M.BoundaryFaces::find(p);		
	switch (example_id) {
	case 0 :
	    break;
	case 1 : 
	    it->second.SetPart(1);
	    break;
	case 2 : 
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
	case 3 :
	    it->second.SetPart(1);
	    break;	    
	case 4 :
	    if (p[0] == 0.0 || p[1] == 0.0 || p[2] == 0.0) {
		it->second.SetPart(1); 
		continue;
	    }
	    else if (p[0] == 1.0) {
		it->second.SetPart(2);
	    }
	    else if (p[1] == 1.0) {
		it->second.SetPart(3);
	    }
	    else if (p[2] == 1.0) {
		it->second.SetPart(4);
	    }
	    break;
	case 5 :
	    break;
	case 6 :
	    break;
	case 7 :
	    break;
	}
    }
}

bool TElasticityProblems::IsDirichlet (int id) {
    switch (example_id) {
    case 0 : {
	if (id == -1) return false;
	if (bc[id-1][0] == 0) {
	    return true;
	}
	return false;
	break;
    }
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
    case 3 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 4 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 5 : {
	if (id == 2 || id == 4) {
	    return true;
	}
	return false;
	break;
    }
    case 6 : {
	if (id == 2) {
	    return true;
	}
	return false;
	break;
    }
    case 7 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    }
}

void TElasticityProblems::g_D (Point p, double time, int k, RowBndValues& u_c, int id) {
    Point pp = zero;
    switch (example_id) {
    case 0 : {
	Point pp;
	pp[0] = bc[id-1][1];
	pp[1] = bc[id-1][2];
	pp[2] = bc[id-1][3];
	u_c.D(k, 0) = true;
	u_c(k, 0) = bc[id-1][1]*time;
	u_c.D(k, 1) = true;
	u_c(k, 1) = bc[id-1][2]*time;
	u_c.D(k, 2) = true;
	u_c(k, 2) = bc[id-1][3]*time;
	break;
    }
    case 1 : { 
	if (id == 1) {
	    pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	    pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = (0.2 * pp * exp(time))[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = (0.2 * pp * exp(time))[1];
	}
	break;
    }
    case 2 : {
	if (id == 1) {
	    pp[0] = sin(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = cos(Pi*p[0]) * sin(Pi*p[1]);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = (0.2 * pp * exp(time))[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = (0.2 * pp * exp(time))[1];
	}
	break;
    }
    case 3 : {
	if (id == 1) {
	    Point pp;
	    pp[0] = exp(p[0] + p[1] + p[2]);
	    pp[1] = exp(p[0] + p[1] + p[2]);
	    pp[2] = exp(p[0] + p[1] + p[2]);
	    pp = 0.01 * pp * exp(time);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = pp[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = pp[1];
	    u_c.D(k, 2) = true;
	    u_c(k, 2) = pp[2];
	}
	break;
    }
    case 4 : {
	if (id == 1) {
	    Point pp;
	    pp[0] = exp(p[0] + p[1] + p[2]);
	    pp[1] = exp(p[0] + p[1] + p[2]);
	    pp[2] = exp(p[0] + p[1] + p[2]);
	    pp = 0.01 * pp * exp(time);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = pp[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = pp[1];
	    u_c.D(k, 2) = true;
	    u_c(k, 2) = pp[2];
	}
	break;
    }
    case 5 : {
	if (id == 4) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = 0;
	}
	else if (id == 2) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = time;
	}
	break;
    }
    case 6 : {
	if (id == 2) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	}
	break;
    }
    case 7 : {
	if (id == 1) {
	    double d = norm(p);
	    u_c(k, 0)   = d*cos(time*1.5707963) - p[0];
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = d*sin(time*1.5707963) - p[1];
	    u_c.D(k, 1) = true;
	}
	break;
    }
    }
}

Point TElasticityProblems::g_N (Point p, double t, int id) {
    Point pp;
    switch (example_id) {
    case 0 : {
	Point pp;
	pp[0] = bc[id-1][1];
	pp[1] = bc[id-1][2];
	pp[2] = bc[id-1][3];
	return pp;
    }
    case 2 : 
	if (id == 2) {
	    pp[0] = 2 * (mu + lambda) * Pi * cos(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = -2 * mu * Pi * sin(Pi*p[0]) * sin(Pi*p[1]);
	    return 0.2 * pp * exp(t);
	}
	else if (id == 3) {
	    pp[0] = -2 * mu * Pi * sin(Pi*p[0]) * sin(Pi*p[1]);
	    pp[1] = 2 * (mu+lambda) * Pi * cos(Pi*p[0]) * cos(Pi*p[1]);
	    return 0.2 * pp * exp(t);
	}
	break;
    case 4 : {
	if (id == 2) {
	    Point pp;
	    pp[0] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    pp[1] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[2] = 2 * mu * exp(p[0] + p[1] + p[2]);
			return 0.01 * pp * exp(t);
	}
	else if (id == 3) {
	    Point pp;
	    pp[0] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[1] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    pp[2] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp * exp(t);
	}
	else if (id == 4) {
	    Point pp;
	    pp[0] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[1] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[2] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp * exp(t);
	}
	break;
    }
    case 5 : {
	if (id == 1 || id == 3) {
	    return zero;
	}
	break;
    }
    case 6 : {
	if (id == 1 || id == 3 || id == 4) {
	    return zero;
	}
	break;
    }
    case 7 : {
	if (id == 2 || id == 3 || id == 4) {
	    return zero;
	}
	break;
    }
    }
}

Point TElasticityProblems::f (Point p, double t) {
    Point pp = zero;
    switch (example_id) {
    case 0 : {
	return zero;
    }
    case 1 : 
	pp[0] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(t);
	break;
    case 2 :
	pp[0] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(t);
	break;
    case 3 : {
	Point pp;
	pp[0] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	pp[1] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	pp[2] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	return 0.01 * pp * exp(t);
    }
    case 4 : {
	Point pp;
	pp[0] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	pp[1] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	pp[2] = (1-3 * (2 * mu + lambda)) * exp(p[0] + p[1]  + p[2]);
	return 0.01 * pp * exp(t);
    }
    case 5 : {
	return zero;
    }
    case 6 : {
	return zero;
    }
    case 7 : {
	return zero;
    }
    }
}

Point TElasticityProblems::u (Point p, double t) {
    Point pp = zero;
    switch (example_id) {
    case 1 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(t);
	break;
    case 2 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(t);
	break;
    case 3 :
        pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
        return 0.01 * pp * exp(t);
        break;
    case 4 :
        pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
	return 0.01 * pp * exp(t);
	break;
    case 5 :
        return zero;
        break;
    case 6 :
        return zero;
        break;
    case 7 :
        return zero;
        break;
    }
}

Point TElasticityProblems::h0 (Point p) {
    Point pp = zero;
    switch (example_id) {
    case 0 :
        return zero;
        break;
    case 1 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp;
	break;
    case 2 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp;
	break;
    case 3 :
        pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
        return 0.01 * pp;
        break;
    case 4 :
        pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
        return 0.01 * pp;
        break;
    case 5 :
        return zero;
        break;
    case 6 :
        return Point(-1.0*time_k,0,0);
        break;
    case 7 :
	return zero;
        break;
    }
}

Point TElasticityProblems::h1 (Point p, double dt) {
    Point pp = zero;
    switch (example_id) {
    case 0 :
	return zero;
	break;
    case 1 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * exp(dt) * pp;
	break;
    case 2 : 
	pp[0] = sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * exp(dt) * pp;
	break;
    case 3 :
	pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
        return 0.01 * pp * exp(dt);
        break;
    case 4 :
        pp[0] = exp(p[0] + p[1] + p[2]);
	pp[1] = exp(p[0] + p[1] + p[2]);
	pp[2] = exp(p[0] + p[1] + p[2]);
        return 0.01 * pp * exp(dt);
        break;
    case 5 :
        return zero;
        break;
    case 6 :
        return zero;
        break;
    case 7 :
	return zero;
        break;
    }
}


