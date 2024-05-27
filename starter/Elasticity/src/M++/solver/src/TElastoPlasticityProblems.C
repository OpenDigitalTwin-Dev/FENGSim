// file: Solution.C
// author: Jiping Xin

#include "TElastoPlasticityProblems.h"

void TElastoPlasticityProblems::SetSubDomain (Mesh& M) {
    for (cell c = M.cells(); c != M.cells_end(); c++) {
	Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case 0 : // fengsim 
	    break;
	case 1 : // 2d elasticity dirichlet b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 2 : // 2d mixed boundary condition 
	    it->second->SetSubdomain(0);
	    break;
	case 3 : // 2d perfect plasticity
	    it->second->SetSubdomain(0);
	    break;
	case 4 : // 3d perfect plasticity
	    it->second->SetSubdomain(0);
	    break;
	case 7 : // 2d abaqus
	    it->second->SetSubdomain(0);
	    break;
	case 8 : // rotation for total lagrange
	    it->second->SetSubdomain(0);
	    break;
	case 9 : // traction test instead of analytical solution for updated lagrange
	    it->second->SetSubdomain(0);
	    break;
	case 10 : // rotation for updated lagrange
	    it->second->SetSubdomain(0);
	    break;
	case 11 : // taylor bar for updated lagrange
	    it->second->SetSubdomain(0);
	    break;
	case 12 : // traction
	    it->second->SetSubdomain(0);
	    break;
	case 13 : // impaction
	    it->second->SetSubdomain(0);
	    break;
	case 14 : // rotation
	    it->second->SetSubdomain(0);
	    break;
	case 15 : // 3d perfect plasticity, mass lumping, dynamic explicit
	    it->second->SetSubdomain(0);
	    break;
	}   
    }
}

void TElastoPlasticityProblems::SetBoundaryType (Mesh& M, double time) {
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
	case 3 : {
	    break;
	}
	case 4 : {
	    break;
	}
	case 7 : {
	    if (p[0] == 0)
		it->second.SetPart(1);
	    else if (p[0] == 1) {
		if (p[1] > 0.75)
		    it->second.SetPart(2);
		else
		    it->second.SetPart(3);
	    }
	    else if (p[1] == 0)
		it->second.SetPart(4);
	    else if (p[1] == 1)
		it->second.SetPart(5);
	    break;
	}
	case 8 : {
	    if (p[1] == 0)
		it->second.SetPart(1);
	    else
		it->second.SetPart(2);
	    break;
	}
	case 9 : {
	    if (p[0] == 0.0) 
		it->second.SetPart(1);
	    else if (abs(p[0] - (1.0+time-time_k))<1e-5) {
		it->second.SetPart(2);
	    }
	    else
		it->second.SetPart(3);
	    break;
	}
	case 10 : {
	    if (abs(p[0]/norm(p) - cos((time-time_k)*1.5707963)) < 1e-5) 
		it->second.SetPart(1);
	    else
		it->second.SetPart(2);
	    break;
	}
	case 11 : {
	    if (abs(p[0] - 1.0)<1e-10) {
		it->second.SetPart(1);
	    }
	    else
		it->second.SetPart(2);
	    break;
	}
	case 12 :
	    break;
	case 13 :
	    break;
	case 14 :
	    break;
	case 15 : {
	    if (abs(p[0] - 1.0)<1e-10) {
		it->second.SetPart(1);
	    }
	    else if (abs(p[0])<1e-10) {
		it->second.SetPart(2);
	    }
	    else
		it->second.SetPart(3);
	    break;
	}
	}
    }
}

bool TElastoPlasticityProblems::IsDirichlet (int id) {
    switch (example_id) {
    case 0 : {
	if (id == -1) return false;
	if (bc[id-1][0] == 0) {
	    return true;
	}
	return false;
	break;
    }
    case 1 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 2 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 3 : {
        if (id == 3 || id == 4) {
	    return true;
	}
	return false;
	break;
    }
    case 4 : {
        if (id == 2 || id == 3 || id == 4) {
	    return true;
	}
	return false;
	break;
    }
    case 7 : {
        if (id == 1 || id == 2 || id == 4) {
	    return true;
	}
	return false;
	break;
    }
    case 8 : {
        if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 9 : {
	if (id == 1 || id == 2) {
	    return true;
	}
	return false;
	break;
    }
    case 10 : {
        if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 11 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 12 : {
	if (id == 2 || id == 4) {
	    return true;
	}
	return false;
	break;
    }
    case 13 : {
	if (id == 2) {
	    return true;
	}
	return false;
	break;
    }
    case 14 : {
	if (id == 1) {
	    return true;
	}
	return false;
	break;
    }
    case 15 : {
	if (id == 1 || id == 2) {
	    return true;
	}
	return false;
	break;
    }
    }
}

void TElastoPlasticityProblems::Dirichlet (Point p, double T, int k, RowBndValues& u_c, int id) {
    Point pp = zero;
    switch (example_id) {
    case 0 : {
	Point pp;
	pp[0] = bc[id-1][1];
	pp[1] = bc[id-1][2];
	pp[2] = bc[id-1][3];
	u_c.D(k, 0) = true;
	u_c(k, 0) = bc[id-1][1]*T;
	u_c.D(k, 1) = true;
	u_c(k, 1) = bc[id-1][2]*T;
	u_c.D(k, 2) = true;
	u_c(k, 2) = bc[id-1][3]*T;
	break;
    }
    case 1 : {
	if (id == 1) {
	    pp[0] = sin(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = cos(Pi*p[0]) * sin(Pi*p[1]);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = (0.2 * pp * exp(T))[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = (0.2 * pp * exp(T))[1];
	}
	break;
    }
    case 2 : { 
	if (id == 1) {
	    pp[0] = sin(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = cos(Pi*p[0]) * sin(Pi*p[1]);
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = (0.2 * pp * exp(T))[0];
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = (0.2 * pp * exp(T))[1];
	}
	break;
    }
    case 3 : 
	if (id == 4) {
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = 0;
	}
	else if (id == 3) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	} 
	break;
    case 4 : 
	if (id == 4) {
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = 0;
	}
	else if (id == 3) {
	    u_c.D(k, 2) = true;
	    u_c(k, 2) = 0;
	} 
	else if (id == 2) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	} 
	break;
    case 7 : {
	if (id == 1) {
	    u_c(k, 0)   = 0.0;
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = 0.0;
	    u_c.D(k, 1) = true;
	}
	else if (id == 2) {
	    u_c(k, 0)   = -0.1 * T;
	    u_c.D(k, 0) = true;
	}
	else if (id == 4) {
	    u_c(k, 0)   = 0.0;
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = 0.0;
	    u_c.D(k, 1) = true;
	}
	break;
    }
    case 8 : {
	if (id == 1) {
	    if (T==0) {
		u_c(k, 0)   = 0;
		u_c.D(k, 0) = true;
		u_c(k, 1)   = 0;
		u_c.D(k, 1) = true;
	    }
	    else {
		u_c(k, 0)   = p[0]*cos(T*1.5707963) - p[0];
		u_c.D(k, 0) = true;
		u_c(k, 1)   = p[0]*sin(T*1.5707963);
		u_c.D(k, 1) = true;
	    }
	}
	break;
    }
    case 9 : {
	if (id == 1) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = 0;
	}
	else if (id == 2) {
	    u_c.D(k, 0) = true;
	    if (!ngeom) {
		u_c(k, 0) = 0.3 * T;
	    }
	    else {
		u_c(k, 0) = 1.0 * time_k;
	    }
	}
	break;
    }
    case 10 : {
	if (id == 1) {
	    double d = norm(p);
	    u_c(k, 0)   = d*cos(T*1.5707963) - d*cos((T-time_k)*1.5707963);
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = d*sin(T*1.5707963) - d*sin((T-time_k)*1.5707963);
	    u_c.D(k, 1) = true;
	}
	break;
    }
    case 11 : {
	if (id == 1) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	}
	break;
    }
    case 12 : {
	if (id == 4) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	    u_c.D(k, 1) = true;
	    u_c(k, 1) = 0;
	}
	else if (id == 2) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = T;
	}
	break;
    }
    case 13 : {
	if (id == 2) {
	    u_c.D(k, 0) = true;
	    u_c(k, 0) = 0;
	}
	break;
    }
    case 14 : {
	if (id == 1) {
	    double d = norm(p);
	    u_c(k, 0)   = d*cos(T*1.5707963) - p[0];
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = d*sin(T*1.5707963) - p[1];
	    u_c.D(k, 1) = true;
	}
	break;
    }
    case 15 : {
	if (id == 1) {
	    u_c(k, 0)   = T;
	    u_c.D(k, 0) = true;
	}
	else if (id == 2) {
	    u_c(k, 0)   = 0;
	    u_c.D(k, 0) = true;
	    u_c(k, 1)   = 0;
	    u_c.D(k, 1) = true;
	    u_c(k, 2)   = 0;
	    u_c.D(k, 2) = true;
	}
	break;
    }
    }
}

Point TElastoPlasticityProblems::Neumann (Point p, double t, int id) {
    Point pp;
    switch (example_id) {
    case 0 : {
	Point pp;
	pp[0] = bc[id-1][1];
	pp[1] = bc[id-1][2];
	pp[2] = bc[id-1][3];
	return pp;
    }
    case 2 : {
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
    }
    case 3 : 
	if (id == 1) {
	    double a = 1.0;
	    double b = 2.0;
	    double c = t;
	    double Y = 0.24 * 2.0 / sqrt(3.0);
	    double r = b;
	    double pa = Y * (log(c / a) + 0.5 * (1.0 - c * c / b / b));
	    return pa * p / norm(p);
	}
	return zero;
	break;
    case 4 : 
        if (id == 5) {
	    double a = 1.0;
	    double b = 2.0;
	    double c = t;
	    double Y = 0.24;
	    double r = b;
	    double pa = Y * (2.0 * log(c / a) + 2.0 / 3.0 * (1.0 - c * c * c / b / b / b));
	    return pa * p / norm(p);
	}
	return zero;
	break;
    case 7 : {
	return zero;
	break;
    }
    case 8 : {
	return zero;
	break;
    }
    case 9 : {
	return zero;
	break;
    }
    case 10 : {
	return zero;
	break;
    }
    case 11 : {
	return zero;
	break;
    }
    case 12 : {
	if (id == 1 || id == 3) {
	    return zero;
	}
	break;
    }
    case 13 : {
	if (id == 1 || id == 3 || id == 4) {
	    return zero;
		}
	break;
    }
    case 14 : {
	if (id == 2 || id == 3 || id == 4) {
	    return zero;
	}
	break;
    }
    case 15 : {
	if (id == 3) {
	    return zero;
	}
	break;
	}
    }
}

Point TElastoPlasticityProblems::Source (Point p, double time) {
    Point pp = zero;
    switch (example_id) {
    case 0 : {
	pp = zero;
	return pp;
	break;
    }
    case 1 : {
	pp[0] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(time);
    }
    case 2 :
	pp[0] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * sin(Pi * p[0]) * cos(Pi * p[1]);
	pp[1] = (1.0 + 2.0 * (2.0 * mu + lambda) * Pi * Pi) * cos(Pi * p[0]) * sin(Pi * p[1]);
	return 0.2 * pp * exp(time);
	break;
    case 3 : 
	pp = zero;
	return pp;
	break;
    case 4 : 
	pp = zero;
	return pp;
	break;
    case 7 : { 
	pp = zero;
	return pp;
	break;
    }
    case 8 : { 
	pp = zero;
	return pp;
	break;
    }
    case 9 : {
	pp = zero;
	return pp;
	break;
    }
    case 10 : {
	pp = zero;
	return pp;
	break;
    }
    case 11 : {
	pp = zero;
	return pp;
	break;
    }
    case 12 : {
	return zero;
    }
    case 13 : {
	return zero;
    }
    case 14 : {
	return zero;
    }
    case 15 : {
	return zero;
    }
    }
}

Point TElastoPlasticityProblems::Solution (Point p, double t) {
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
	pp = zero;
	return pp;
	break;
    case 4 : 
	pp = zero;
	return pp;
	break;
    case 7 : { 
	pp = zero;
	return pp;
	break;
    }
    case 8 : { 
	pp = zero;
	return pp;
	break;
    }
    case 9 : { 
	pp = zero;
	return pp;
	break;
    }
    case 10 : { 
	pp = zero;
	return pp;
	break;
    }
    case 11 : { 
	pp = zero;
	return pp;
	break;
    }
    case 12 :
        return zero;
        break;
    case 13 :
        return zero;
        break;
    case 14 :
	return zero;
	break;
    case 15 :
	return zero;
	break;
    }
}

Point TElastoPlasticityProblems::h0 (Point p) {
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
    case 7 :
        return zero;
        break;
    case 8 :
        return zero;
        break;
    case 9 :
        return zero;
        break;
    case 10 :
        return zero;
        break;
    case 11 :
	pp[0] = 1.0 * -1.0 * time_k;
	pp[1] = 0;
	pp[2] = 0;
        return pp;
        break;
    case 12 :
        return zero;
        break;
    case 13 :
        return Point(-1.0*time_k,0,0);
        break;
    case 14 :
        return zero;
        break;
    case 15 :
        return zero;
        break;
    }
}

Point TElastoPlasticityProblems::h1 (Point p, double dt) {
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
    case 7 :
        return zero;
        break;
    case 8 :
        return zero;
        break;
    case 9 :
        return zero;
        break;
    case 10 :
        return zero;
        break;
    case 11 :
        pp[0] = 0;
	pp[1] = 0;
	pp[2] = 0;
        return pp;
        break;
    case 12 :
        return zero;
        break;
    case 13 :
        return zero;
        break;
    case 14 :
	return zero;
        break;
    case 15 :
	return zero;
        break;
    }
}

bool TElastoPlasticityProblems::Contact (Point x, Point x1, Point x2, Point x3, Point x4) {
    double area = norm(x2-x1) * norm(x4-x1);
    double a,b,c,s;
    double area_sum = 0;
    a = norm(x-x1);
    b = norm(x-x2);
    c = norm(x1-x2);
    s = 0.5 * (a + b + c);
    area_sum += sqrt(s * (s-a) * (s-b) * (s-c));
    a = norm(x-x2);
    b = norm(x-x3);
    c = norm(x2-x3);
    s = 0.5 * (a + b + c);
    area_sum += sqrt(s * (s-a) * (s-b) * (s-c));
    a = norm(x-x3);
    b = norm(x-x4);
    c = norm(x3-x4);
    s = 0.5 * (a + b + c);
    area_sum += sqrt(s * (s-a) * (s-b) * (s-c));
    a = norm(x-x4);
    b = norm(x-x1);
    c = norm(x4-x1);
    s = 0.5 * (a + b + c);
    area_sum += sqrt(s * (s-a) * (s-b) * (s-c));
    if (area_sum == area) return true;
    else return false;
}

bool TElastoPlasticityProblems::Contact (Point x, Point x1, Point x2) {
    double length = norm(x2-x1);
    double length_sum = norm(x-x1) + norm(x-x2);
    if (length_sum == length) return true;
    else return false;
}
