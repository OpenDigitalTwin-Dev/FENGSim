// file:   FaceElement.h
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/FaceElement.C,v 1.6 2009-01-09 17:57:53 mueller Exp $

#include "FaceElement.h" 

FaceElement::FaceElement (const matrixgraph& g, const cell& c, int face, 
			  const Quadrature& Quad) 
	: rows(g,c,face), Q(Quad) {
    double ww = 0;
    for (int q=0; q<Q.size(); ++q) ww += Q.Weight(q);
    ww = c.LocalFaceArea(face) / ww;
    for (int q=0; q<Q.size(); ++q) {
        const Point& x = Q.QPoint(q);
        Point y = zero;
        switch (c.FaceCorners(face)) {
        case 2:
            y = (1-x[0])*c.LocalCorner(c.facecorner(face,0))
                +  x[0] *c.LocalCorner(c.facecorner(face,1));
            break;
        case 3:
            y = (1-x[0]-x[1])*c.LocalCorner(c.facecorner(face,0))
                +       x[0] *c.LocalCorner(c.facecorner(face,1))
                +       x[1] *c.LocalCorner(c.facecorner(face,2));
            break;
        case 4:
            y =  (1-x[0])*(1-x[1])*c.LocalCorner(c.facecorner(face,0))
                +   x[0] *(1-x[1])*c.LocalCorner(c.facecorner(face,1))
                +   x[0] *   x[1] *c.LocalCorner(c.facecorner(face,2))
                +(1-x[0])*   x[1] *c.LocalCorner(c.facecorner(face,3));
            break;
        }
        qLocal[q] = y;
        qPoint[q] = c[y]; 
	
        T[q] = c.GetTransformation(y);
	
        qNormal[q] = T[q] * c.LocalFaceNormal(face);
	
	double w = norm(qNormal[q]);
        qWeight[q] = T[q].Det() * w * ww * Q.Weight(q);

        qNormal[q] *= (1/w);
    }
}

double FaceElement::Area () const {
    double a = 0;
    for (int q=0; q<Q.size(); ++q) 
	a += qWeight[q];
    return a;
}

ScalarFaceElement::ScalarFaceElement (const Discretization& D, const matrixgraph& g,
                                      const cell& c, int face, int p)
	: FaceElement(g,c,face,D.GetFaceQuad(c,face)), S(D.GetShape(c,p)) {
    for (int q=0; q<nQ(); ++q) {
        for (int i=0; i<size(); ++i) {
            Point Q = QLocal(q);
            int j   = g.NodalPointOnFace(c,face,i);
            value[q][i]    = S(Q,j);
            gradient[q][i] = GetTransformation(q) * S.LocalGradient(Q,j);
        }
    }
}

Scalar ScalarFaceElement::Value (int q, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size(); ++i) {
        U += u(r(i),k) * value[q][i];
    }
    return U;
}

ostream& operator << (ostream& s, const ScalarFaceElement& E) {
    return s << E.S; }

CurlFaceElement::CurlFaceElement (const Discretization& D, const matrixgraph& g,
				  const cell& c, int face, int p)
	: FaceElement(g,c,face,D.GetFaceQuad(c,face)), S(D.GetShape(c,p)) {
    for (int q=0; q<nQ(); ++q) {
        Point Q = QLocal(q);
        double idet = 1/GetTransformationDet(q);
        for (int i=0; i<size(); ++i) {
            int j = g.NodalPointOnFace(c,face,i);
	    double sign = 2 * (c.EdgeCorner(j,0) < c.EdgeCorner(j,1)) - 1; 
	    vectorfield[q][i] = sign * 
	        (GetTransformation(q) * S.LocalVector(Q,j));
	    curlfield[q][i] = sign * idet * 
		GetTransformation(q).ApplyJ(S.LocalCurl(Q,j));
        }
    }
}

VectorField VectorFieldFaceElement::VectorValue (int q, int i, int k) const {
    VectorField V = 0.0;
    V[k] = Value(q,i);
    return V;
}

VectorField VectorFieldFaceElement::VectorValue (int q, const Vector& u) const {
    VectorField V = 0.0;
    for (int i=0; i<size(); ++i) {
        Scalar s = Value(q,i);
        for (int k=0; k<dim; ++k) 
            V[k] += s * u(r(i),k);
    }
    return V;
}

MixedFaceElement::MixedFaceElement (const Discretization& D, const matrixgraph& g,
				    const cell& c, int face)
    : FaceElement(g,c,face,D.GetFaceQuad(c,face)),
      S1(D.GetShape(c,0)),
      S2(D.GetShape(c,1)),
      C(c),
      dim(c.dim()) {

    if (D.DiscName()=="CosseratP1") {
	size_1 = c.Corners();
	size_2 = c.Corners();
    }
    else if (D.DiscName()=="CosseratP2") {
	size_1 = c.Corners()+c.Edges();
	size_2 = c.Corners()+c.Edges();
    }
    else if (D.DiscName()=="CosseratP2P1") {
	size_1 = c.Corners()+c.Edges();
	size_2 = c.Corners();
    }
    else {
	size_1 = D.NodalPoints(c);
	size_2 = c.Corners();
    }
    
    for (int q=0; q<nQ(); ++q) {
        for (int i=0; i<size_1; ++i) {
            Point Q = QLocal(q);
            int j   = g.NodalPointOnFace(c,face,i);
            value1[q][i]    = S1(Q,j);
            gradient1[q][i] = GetTransformation(q) * S1.LocalGradient(Q,j);
        }
        for (int i=0; i<size_2; ++i) {
            Point Q = QLocal(q);
            int j   = g.NodalPointOnFace(c,face,i);
            value2[q][i]    = S2(Q,j);
            gradient2[q][i] = GetTransformation(q) * S2.LocalGradient(Q,j);
        }
    }
}

Gradient MixedFaceElement::Derivative_1 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_1; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S1.LocalGradient(z,i));
    }
    return Du;
}

Scalar MixedFaceElement::Value_2 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_2; ++i) U += u(r(i),k) * S2(z,i);
    return U;
}

ostream& operator << (ostream& s, const MixedFaceElement& E) {
    return s << E.S1 << E.S2; 
}

//---------------------------------------------------------------------
// RTFaceElement  -  will only work for lowest order RT, since only
// one dof per edge is assumed
//---------------------------------------------------------------------


RT0FaceElement::RT0FaceElement (const Discretization& D, const matrixgraph& g,
		   const cell& c, int face, int p) 
    : FaceElement(g,c,face,D.GetFaceQuad(c,face)), S(D.GetShape(c,p)) {
    Transformation T = c.GetTransformation( c.Face(face) );
    normal =  T * c.LocalFaceNormal(face);
    double dummy = norm(normal);
    if ( ( (*this)[0]() + normal) > (*this)[0]() )
	sign = 1;
    else sign = -1;
    normal *= ( sign/dummy);
    for (int q=0; q<nQ(); ++q) {
//	for (int i=0; i<size(); ++i) {  // nodal points
//	mout << "dummy " <<dummy<<endl;
//	    normal[face] = (1.0/dummy)*normal[face];
	vectorfield[q] = sign/T.Det()
	    *T.ApplyJ(S.LocalVector(QLocal(q),face));
    }
}
VectorField RT0FaceElement::VectorValue (int q, int k) const {
    return vectorfield[q];
}

// not tested yet
VectorField RT0FaceElement::VectorValue (int q, const Vector& u, 
					 int k) const {
    VectorField V=0.0;
    return u(r(0),k)*vectorfield[q];
}

