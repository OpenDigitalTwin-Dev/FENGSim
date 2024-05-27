// file: Solution.C
// author: Jiping Xin

#include "ElasticityProblems.h"

void ElasticityProblems::SetSubDomain (Mesh& M) {
    int example_id = 1;
    ReadConfig(Settings, "EXAMPLE", example_id);
    for (cell c = M.cells(); c != M.cells_end(); c++) {
        Point p(c());
	hash_map<Point,Cell*,Hash>::iterator it = M.Cells::find(p);
	switch (example_id) {
	case 0 : // fengsim 
	    break;
	case 1 : // 2d elasticity dirichlet b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 2 : // 2d elasticity mixed b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 3 : // 3d elasticity dirichlet b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 4 : // 3d elasticity mixed b.c. 
	    it->second->SetSubdomain(0);
	    break;
	case 5 : // cook membrane
	    it->second->SetSubdomain(0);
	    break;
	}
    }
}

void ElasticityProblems::SetBoundaryType (Mesh& M) {
    int example_id = 1;
    ReadConfig(Settings, "EXAMPLE", example_id);
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
	    }
	    else if (p[1] == 1.0) {
		it->second.SetPart(3);
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
	}
    }
}

bool ElasticityProblems::IsDirichlet (int id) {
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
	if (id == 4) {
	    return true;
	}
	return false;
	break;
    }
    }
    return false;
}

Point ElasticityProblems::g_D (int k, RowBndValues& u_c, int id, Point p) {
    switch (example_id) {
    case 0 : {
	Point pp;
	pp[0] = bc[id-1][1];
	pp[1] = bc[id-1][2];
	pp[2] = bc[id-1][3];
	return pp;
	break;
    }
    case 1 : {
        if (id == 1) {
	    Point pp;
	    pp[0] = sin(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = cos(Pi*p[0]) * sin(Pi*p[1]);
	    return 0.2 * pp;
	}
	break;
    }
    case 2 : {
	if (id == 1) {
	    Point pp;
	    pp[0] = sin(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = cos(Pi*p[0]) * sin(Pi*p[1]);
	    return 0.2 * pp;
	}
	break;
    }
    case 3 : {
	if (id == 1) {
	    Point pp;
	    pp[0] = exp(p[0] + p[1] + p[2]);
	    pp[1] = exp(p[0] + p[1] + p[2]);
	    pp[2] = exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp;
	}
	break;
    }
    case 4 : {
	if (id == 1) {
	    Point pp;
	    pp[0] = exp(p[0] + p[1] + p[2]);
	    pp[1] = exp(p[0] + p[1] + p[2]);
	    pp[2] = exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp;
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
	break;
    }
    }
}

Point ElasticityProblems::g_N (int id, Point p) {
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
	    Point pp;
	    pp[0] = 2 * (mu + lambda) * Pi * cos(Pi*p[0]) * cos(Pi*p[1]);
	    pp[1] = -2 * mu * Pi * sin(Pi*p[0]) * sin(Pi*p[1]);
	    return 0.2 * pp;
	}
	else if (id == 3) {
	    Point pp;
	    pp[0] = -2 * mu * Pi * sin(Pi*p[0]) * sin(Pi*p[1]);
	    pp[1] = 2 * (mu + lambda) * Pi * cos(Pi*p[0]) * cos(Pi*p[1]);
	    return 0.2 * pp;
	}
	break;
    }
    case 4 : {
	if (id == 2) {
	    Point pp;
	    pp[0] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    pp[1] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[2] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp;
	}
	else if (id == 3) {
	    Point pp;
	    pp[0] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[1] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    pp[2] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp;
	}
	else if (id == 4) {
	    Point pp;
	    pp[0] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[1] = 2 * mu * exp(p[0] + p[1] + p[2]);
	    pp[2] = (2 * mu + 3 * lambda) * exp(p[0] + p[1] + p[2]);
	    return 0.01 * pp;
	}
	break;
    }
    case 5 : {
	if (id == 1) {
	    return zero;
	}
	else if (id == 2) {
	    return Point(0, 1.0/16.0, 0);
	}
	else if (id == 3) {
	    return zero;
	}
	break;
    }
    }
}
    
Point ElasticityProblems::f (int id, Point p) {
    switch (example_id) {
    case 0 : {
	return zero;
    }
    case 1 : {
	if (id == 0) {
	    Point pp;
	    pp[0] = 2 * (2 * mu + lambda) * Pi * Pi * sin(Pi * p[0]) * cos(Pi * p[1]);
	    pp[1] = 2 * (2 * mu + lambda) * Pi * Pi * cos(Pi * p[0]) * sin(Pi * p[1]);
	    return 0.2 * pp;
	}
    }
    case 2 : {
	if (id == 0) {
	    Point pp;
	    pp[0] = 2 * (2 * mu + lambda) * Pi * Pi * sin(Pi * p[0]) * cos(Pi * p[1]);
	    pp[1] = 2 * (2 * mu + lambda) * Pi * Pi * cos(Pi * p[0]) * sin(Pi * p[1]);
	    return 0.2 * pp;
	}
    }
    case 3 : {
	if (id == 0) {
	    Point pp;
	    pp[0] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    pp[1] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    pp[2] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    return 0.01 * pp;
	}
    }
    case 4 : {
	if (id == 0) {
	    Point pp;
	    pp[0] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    pp[1] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    pp[2] = -3 * (2 * mu + lambda) * exp(p[0] + p[1]  + p[2]);
	    return 0.01 * pp;
	}
    }
    case 5 : {
	if (id == 0) {
	    Point pp;
	    pp = zero;
	    return pp;
	}
    }
    }
    return zero;
}

Point ElasticityProblems::u (Point p) {
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
        pp = zero;
        return pp;
        break;
    }
}

