// file: Element.C
// author: Christian Wieners
// $Header: /public/M++/src/Element.C,v 1.14 2009-10-21 09:58:56 wieners Exp $

#include "Element.h"

Element::Element (const matrixgraph& g, const cell& c,
                  const Quadrature& Quad) : rows(g,c), Q(Quad) {
    for (int q=0; q<Q.size(); ++q) {
        T[q] = c.GetTransformation(Q.QPoint(q));
        qWeight[q] = T[q].Det() * Q.Weight(q);
        qPoint[q] = c[Q.QPoint(q)]; 
    }
}

ScalarElement::ScalarElement(const Discretization& D,const matrixgraph& g,const cell& c) :
    Element(g,c,D.GetQuad(c)), S(D.GetShape(c)), value(S.values()), C(c) {
    for (int q=0; q<nQ(); ++q) 
		for (int i=0; i<size(); ++i) { 
// Let F = Grad(phi) be the gradient of the transformation 
// from the reference element to the element Then the following return F^{-T} Grad(N)
// where N is a shape function
			gradient[q][i] = GetTransformation(q) * S.LocalGradient(q,i);
// whereas the following would return  F^{-1} Grad(N)
//            gradient[q][i] = S.LocalGradient(q,i) * GetTransformation(q);
		}
}

Scalar ScalarElement::Value (int q, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size(); ++i) U += u(r(i),k) * value[q][i];
    return U;
}

Scalar ScalarElement::Value (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size(); ++i) U += u(r(i),k) * S(z,i);





    




    
    
    
    return U;
}

Gradient ScalarElement::Derivative (int q, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size(); ++i) Du += u(r(i),k) * gradient[q][i];
    return Du;
}

Gradient ScalarElement::Derivative (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size(); ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S.LocalGradient(z,i));
    }
    return Du;
}

ostream& operator << (ostream& s, const ScalarElement& E) {
    return s << E.S; }

VectorFieldElement::VectorFieldElement (const Discretization& D, const matrixgraph& g,
			const cell& c) : ScalarElement(D,g,c), dim(c.dim()) { }


VectorField VectorFieldElement::VectorValue (int q, int i, int k) const {
    VectorField V = zero;
    V[k] = Value(q,i);
    return V;
}

VectorField VectorFieldElement::VectorValue (int q, const Vector& u) const {
    VectorField V = zero;
    for (int i=0; i<size(); ++i) {
        Scalar s = Value(q,i);
        for (int k=0; k<dim; ++k) 
            V[k] += s * u(r(i),k);
    }
    return V;
}

VectorField VectorFieldElement::VectorValue (const Point& L,  const Vector& u) const {
    VectorField V = zero;
    for (int k=0; k<dim; ++k) {
        Scalar s = Value(L,u,k);
        V[k] = s;
    }
    return V;
}

Tensor VectorFieldElement::VectorGradient (int q, int i, int k) const {
    Tensor T = Zero;
    T[k] = Derivative(q,i);
    return T;
}

Tensor VectorFieldElement::VectorGradient (int q,  const Vector& u) const {
    Tensor T = Zero;
    for (int i=0; i<size(); ++i) {
        for (int k=0; k<dim; ++k) {
            Gradient G = u(r(i),k) * Derivative(q,i);
            for (int l=0; l<dim; ++l) {
                T[k][l] += G[l];
            }
        }
    }
    return T;
}

Tensor VectorFieldElement::VectorGradient (const Point& L,  const Vector& u) const {
    Tensor T = Zero;
    for (int k=0; k<dim; ++k) {
        Gradient G = Derivative(L,u,k);
        for (int l=0; l<dim; ++l) {
            T[k][0] = G[0];
            T[k][1] = G[1];
            T[k][2] = G[2];
        }
    }
    return T;
}

Scalar VectorFieldElement::Divergence (int q, int i, int k) const {
    Gradient G = Derivative(q,i);
    return G[k];
}

Scalar VectorFieldElement::Divergence (int q, const Vector& u) const {
    Scalar s = 0.0;
    for (int i=0; i<size(); ++i) {
        Gradient G = Derivative(q,i);
        for (int k=0; k<dim; ++k) 
            s += u(r(i),k) * G[k];
    }
    return s;
}


int FaceOrientation (const cell& c, int f) {
    return 2 * ((f==0)?(c[0] > c[1]):(c.FaceCorner(f,1) > c.FaceCorner(f,0))) - 1;
}

Point CurlElement::DofShape_int(double x, double y, double z, int i) const {
       // functions for DOF calculation of Curl2DoF
    switch(i) {
       // 2-nd order face based
       // face (1, 0, 3, 2)
       case(24): return Point(1.5,0,0);
       case(25): return Point(0,-1.5,0);
       case(26): return Point(2.25-4.5*x,-2.25+4.5*y,0);
       case(27): return Point(2.25-4.5*x,2.25-4.5*y);
       // face (0, 1, 5, 4)
       case(28): return Point(-1.5,0,0);
       case(29): return Point(0,0,-1.5);
       case(30): return Point(2.25-4.5*x,0,-2.25+4.5*z);
       case(31): return Point(2.25-4.5*x,0,2.25-4.5*z);
       // face (1, 2, 6, 5)
       case(32): return Point(0,-1.5,0);
       case(33): return Point(0,0,-1.5);
       case(34): return Point(0,2.25-4.5*y,-2.25+4.5*z);
       case(35): return Point(0,2.25-4.5*y,2.25-4.5*z);
       // face (2, 3, 7, 6)
       case(36): return Point(1.5,0,0);
       case(37): return Point(0,0,-1.5);
       case(38): return Point(2.25-4.5*x,0,-2.25+4.5*z);
       case(39): return Point(2.25-4.5*x,0,2.25-4.5*z);
       // face (3, 0, 4, 7)
       case(40): return Point(0,1.5,0);
       case(41): return Point(0,0,-1.5);
       case(42): return Point(0,2.25-4.5*y,-2.25+4.5*z);
       case(43): return Point(0,2.25-4.5*y,2.25-4.5*z);
       // face (4, 5, 6, 7)
       case(44): return Point(-1.5,0,0);
       case(45): return Point(0,-1.5,0);
       case(46): return Point(2.25-4.5*x,-2.25+4.5*y,0);
       case(47): return Point(2.25-4.5*x,2.25-4.5*y,0);
       // 2-nd order cell-based
       case(48): return Point(9,0,0);
       case(49): return Point(0,9,0);
       case(50): return Point(0,0,9);
       case(51): return Point(-6.75+13.5*x,6.75-13.5*y,0);
       case(52): return Point(-6.75+13.5*x,0,6.75-13.5*z);
       case(53): return Point(0,-6.75 + 13.5*y,-6.75 + 13.5*z);
    }
}

int EdgeOrientation (const cell& c, int e) {
    if (c.EdgeCorner(e,0) < c.EdgeCorner(e,1)) 
	return 1;
    return -1;
}

CurlElement::CurlElement (const Discretization& D,const matrixgraph& g,const cell& _c) :
	Element(g,_c,D.GetQuad(_c)), c(_c), S(D.GetShape(_c))  {
    for (int i=0; i<c.Edges(); ++i) {
       sign[i] = 2 * (c.EdgeCorner(i,0) < c.EdgeCorner(i,1)) - 1; 
       tangent[i] = sign[i] * (c.EdgeCorner(i,1) - c.EdgeCorner(i,0));
    }
    if (g.Name() == "Curl2DoF") {
       int n = c.Edges();
       for (int i=n; i<size(); ++i) sign[i] = 1;
       for (int i=0; i<c.Faces(); ++i) sign[2*n+4*i] = FaceOrientation(c,i);
    }
    for (int q=0; q<nQ(); ++q) {
        double idet = 1/GetTransformationDet(q);
        for (int i=0; i<size(); ++i) {
	    vectorfield[q][i] = sign[i] * 
		(GetTransformation(q) * S.LocalVector(LocalQPoint(q),i));
	    curlfield[q][i] = sign[i] * idet * 
		GetTransformation(q).ApplyJ(S.LocalCurl(LocalQPoint(q),i));
        }
    }
}

VectorField CurlElement::VectorValue (const Point& z, const Vector& u, 
				      int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size(); ++i) 
	V += u(r(i),k) * sign[i]*(c.GetTransformation(z) * S.LocalVector(z,i));
    return V;
}

VectorField CurlElement::CurlVector (const Point& z, const Vector& u, 
				      int k) const {
    VectorField V = 0.0;
    double idet = 1/c.GetTransformation(z).Det();
    for (int i=0; i<size(); ++i) 
        V += u(r(i),k) * sign[i] * idet * (c.GetTransformation(z).ApplyJ(S.LocalCurl(z,i)));
    return V;
}

VectorField CurlElement::VectorValue (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size(); ++i) V += u(r(i),k) * vectorfield[q][i];
    return V;
}

VectorField CurlElement::CurlVector (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size(); ++i) V += u(r(i),k) * curlfield[q][i];
    return V;
}

ostream& operator << (ostream& s, const CurlElement& E) {
    return s << E.S; }

/***************************************************************************
 * DivElement
***************************************************************************/
/*DivElement::DivElement (const Discretization& D,const matrixgraph& g,const cell& _c)
    : Element(g,_c,D.GetQuad(_c)), c(_c), S(D.GetShape(_c)) {
    for (int q=0; q<nQ(); ++q) {
	vectorfield[q][i] = (1.0/(GetTransformation(q).Det()))*GetTransformation(q) 
	    * (S.LocalVector(LocalQPoint(q),i));
    }
}
VectorField DivElement::VectorValue (int q, int i) const {
    return vectorfield[q][i];
}


VectorField DivElement::VectorValue (int q, const Vector& u, int k) const {
    VectorField V =0.0;
    for (int i=0; i<size(); ++i) V+= u(r(i),k) * vectorfield[q][i];
    return V;
}

//Scalar DivElement::DivValue (int q,const Vector& u, int i) const;


ostream& operator << (ostream& s, const DivElement& E) {
    return s << E.S; 
}

*/

/***************************************************************************
 * MagnetostaticsMixedElement
***************************************************************************/

QuasiMagnetoStaticsMixedElement::QuasiMagnetoStaticsMixedElement (const Discretization& D, const matrixgraph& g,const cell& c) :
    Element(g,c,D.GetQuad(c)), 
    S1(D.GetShape(c,0)), value(S1.values()),
    S2(D.GetShape(c,1)), 
    C(c),
    dim(c.dim()) {
    size_1 = c.Corners();
    size_2 = c.Edges();

    // scalar element
    for (int q=0; q<nQ(); ++q) 
        for (int i=0; i<c.Corners(); ++i) 
	    gradient[q][i] = GetTransformation(q) * S1.LocalGradient(q,i);

    // vector element
    for (int i=0; i<c.Edges(); ++i) {
        sign[i] = 2 * (c.EdgeCorner(i,0) < c.EdgeCorner(i,1)) - 1; 
	tangent[i] = sign[i] * (c.EdgeCorner(i,1) - c.EdgeCorner(i,0));
    }
    for (int q=0; q<nQ(); ++q) {
        double idet = 1/GetTransformationDet(q);
        for (int i=0; i<c.Edges(); ++i) {
	    vectorfield[q][i] = sign[i] * 
		(GetTransformation(q) * S2.LocalVector(LocalQPoint(q),i));
	    curlfield[q][i] = sign[i] * idet * 
		GetTransformation(q).ApplyJ(S2.LocalCurl(LocalQPoint(q),i));
        }
    }
}

Scalar QuasiMagnetoStaticsMixedElement::Value (int q, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_1; ++i) U += u(r(i),k) * value[q][i];
    return U;
}

Gradient QuasiMagnetoStaticsMixedElement::Derivative (int q, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_1; ++i) Du += u(r(i),k) * gradient[q][i];
    return Du;
}

Gradient QuasiMagnetoStaticsMixedElement::Derivative (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_1; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S1.LocalGradient(z,i));
    }
    return Du;
}

VectorField QuasiMagnetoStaticsMixedElement::VectorValue (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size_2; ++i) V += u(r(i+size_1),k) * vectorfield[q][i];
    return V;
}

VectorField QuasiMagnetoStaticsMixedElement::VectorValue (const Point& z, const Vector& u, 
							  int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size_2; ++i) 
	V += u(r(i+size_1),k) * sign[i]*(C.GetTransformation(z) * S2.LocalVector(z,i));
    return V;
}

VectorField QuasiMagnetoStaticsMixedElement::CurlVector (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size_2; ++i) V += u(r(i+size_1),k) * curlfield[q][i];
    return V;
}

VectorField QuasiMagnetoStaticsMixedElement::CurlVector (const Point& z, const Vector& u, 
				      int k) const {
    VectorField V = 0.0;
    double idet = 1/C.GetTransformation(z).Det();
    for (int i=0; i<size_2; ++i) 
        V += u(r(i+size_1),k) * sign[i] * idet * (C.GetTransformation(z).ApplyJ(S2.LocalCurl(z,i)));
    return V;
}

/***************************************************************************
 * MixedElement
***************************************************************************/

MixedElement::MixedElement (const Discretization& D, const Vector& u, 
			    const cell& c, int _size_1, int _size_2) :
    Element(u,c,D.GetQuad(c)), 
    S1(D.GetShape(c,0)), value1(S1.values()),
    S2(D.GetShape(c,1)), value2(S2.values()),
    dim(c.dim()),
    C(c),
    size_1(_size_1), size_2(_size_2) {
    
    for (int q=0; q<nQ(); ++q) {
        for (int i=0; i<size_1; ++i) { 
	    gradient1[q][i] = GetTransformation(q) * S1.LocalGradient(q,i);
	}
	for (int i=0; i<size_2; ++i) {
	    gradient2[q][i] = GetTransformation(q) * S2.LocalGradient(q,i);
	}
    }
}

MixedElement::MixedElement (const Discretization& D, const matrixgraph& g,const cell& c) :
    Element(g,c,D.GetQuad(c)), 
    S1(D.GetShape(c,0)), value1(S1.values()),
    S2(D.GetShape(c,1)), value2(S2.values()),
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
	for (int i=0; i<size_1; ++i) 
	    gradient1[q][i] = GetTransformation(q) * S1.LocalGradient(q,i);
	for (int i=0; i<size_2; ++i) 
	    gradient2[q][i] = GetTransformation(q) * S2.LocalGradient(q,i);
    }
}

ostream& operator << (ostream& s, const MixedElement& E) {
    return s << E.S1 << E.S2; 
}

Gradient MixedElement::Derivative_1 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_1; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S1.LocalGradient(z,i));
    }
    return Du;
}

Gradient MixedElement::Derivative_2 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_2; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S2.LocalGradient(z,i));
    }
    return Du;
}

Scalar MixedElement::Value_1 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_1; ++i) U += u(r(i),k) * S1(z,i);
    return U;
}

Scalar MixedElement::Value_2 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_2; ++i) U += u(r(i),k) * S2(z,i);
    return U;
}


/***************************************************************************
 * MixedElementPPP
***************************************************************************/

MixedElementPPP::MixedElementPPP (const Discretization& D, const Vector& u, 
			    const cell& c, int _size_1, int _size_2, int _size_3) :
    Element(u,c,D.GetQuad(c)), 
    S1(D.GetShape(c,0)), value1(S1.values()),
    S2(D.GetShape(c,1)), value2(S2.values()),
    S3(D.GetShape(c,2)), value3(S3.values()),
    dim(c.dim()),
    C(c),
    size_1(_size_1), size_2(_size_2), size_3(_size_3) {
    
    for (int q=0; q<nQ(); ++q) {
        for (int i=0; i<size_1; ++i) { 
	    gradient1[q][i] = GetTransformation(q) * S1.LocalGradient(q,i);
	}
	for (int i=0; i<size_2; ++i) {
	    gradient2[q][i] = GetTransformation(q) * S2.LocalGradient(q,i);
	}
	for (int i=0; i<size_3; ++i) {
	    gradient3[q][i] = GetTransformation(q) * S3.LocalGradient(q,i);
	}
    }
}

MixedElementPPP::MixedElementPPP (const Discretization& D, const matrixgraph& g,const cell& c) :
    Element(g,c,D.GetQuad(c)), 
    S1(D.GetShape(c,0)), value1(S1.values()),
    S2(D.GetShape(c,1)), value2(S2.values()),
    S3(D.GetShape(c,2)), value3(S3.values()),
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
	size_3 = c.Corners();
    }
    for (int q=0; q<nQ(); ++q) {
	for (int i=0; i<size_1; ++i) 
	    gradient1[q][i] = GetTransformation(q) * S1.LocalGradient(q,i);
	for (int i=0; i<size_2; ++i) 
	    gradient2[q][i] = GetTransformation(q) * S2.LocalGradient(q,i);
	for (int i=0; i<size_3; ++i) 
	    gradient3[q][i] = GetTransformation(q) * S3.LocalGradient(q,i);
    }
}

ostream& operator << (ostream& s, const MixedElementPPP& E) {
    return s << E.S1 << E.S2; 
}

Gradient MixedElementPPP::Derivative_1 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_1; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S1.LocalGradient(z,i));
    }
    return Du;
}

Gradient MixedElementPPP::Derivative_2 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_2; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S2.LocalGradient(z,i));
    }
    return Du;
}

Gradient MixedElementPPP::Derivative_3 (const Point& z, const Vector& u, int k) const {
    Gradient Du = 0.0;
    for (int i=0; i<size_2; ++i) {
        Transformation lT = C.GetTransformation(z);
        Du += u(r(i),k) * (lT * S3.LocalGradient(z,i));
    }
    return Du;
}

Scalar MixedElementPPP::Value_1 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_1; ++i) U += u(r(i),k) * S1(z,i);
    return U;
}

Scalar MixedElementPPP::Value_2 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_2; ++i) U += u(r(i),k) * S2(z,i);
    return U;
}

Scalar MixedElementPPP::Value_3 (const Point& z, const Vector& u, int k) const {
    Scalar U = 0.0;
    for (int i=0; i<size_3; ++i) U += u(r(i),k) * S3(z,i);
    return U;
}



/***************************************************************************
 * TaylorHoodElement
***************************************************************************/
TaylorHoodElement::TaylorHoodElement (const Discretization& D, const Vector& u, 
				      const cell& c) 
    : MixedElement(D,u,c,c.Corners(),D.NodalPoints(c))  { }
Scalar TaylorHoodElement::PressureValue (int q, int i) const {
    return Value_1(q,i); 
}
Gradient TaylorHoodElement::PressureGradient (int q, int i) const { 
    return Derivative_1(q,i); 
}
Gradient TaylorHoodElement::VelocityField (int q, int i, int k) const {
    Velocity V = 0.0;
    V[k] = Value_2(q,i);
    return V;
}
VelocityGradient TaylorHoodElement::VelocityFieldGradient(int q, int i, int k) const {
    return VelocityGradient( Derivative_2(q,i),k);
}
Scalar TaylorHoodElement::PressureValue (int q, const Vector& u) const { 
    Scalar p = 0.0;
    for (int i=0; i<Size_1(); ++i) p += u(r(i),Dim()) * Value_1(q,i);
    return p;
}
Gradient TaylorHoodElement::PressureGradient (int q, const Vector& u) const { 
    Gradient Dp = 0.0;
    for (int i=0; i<Size_1(); ++i) Dp += u(r(i),Dim()) * Derivative_1(q,i);
    return Dp;
}
Velocity TaylorHoodElement::VelocityField (int q, const Vector& u) const { 
    Velocity V = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) 
	    V[k] += u(r(i),k) * Value_2(q,i);
    return V;
}
VelocityGradient TaylorHoodElement::VelocityFieldGradient (int q, const Vector& u) const { 
    VelocityGradient DV = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) {
	    Gradient G = u(r(i),k) * Derivative_2(q,i);
	    for (int l=0; l<Dim(); ++l)
		DV[k][l] += G[l];
	}
    return DV;
}
Scalar TaylorHoodElement::Divergence (int q, const Vector& u) const {
    Scalar s=0.0;
    for (int i=0; i<Size_2(); ++i)
	for (int k=0; k<Dim(); ++k)
	    s += u(r(i),k) * Derivative_2(q,i)[k];
    return s;
}
Scalar TaylorHoodElement::Divergence (int q, int i, int k) const {
    return Derivative_2(q,i)[k];
}


/***************************************************************************
 * EqualOrderElement (P1P1 or P2P2)
***************************************************************************/
EqualOrderElement::EqualOrderElement (const Discretization& D, const Vector& u, 
				      const cell& c) 
    : MixedElement(D,u,c,D.NodalPoints(c),D.NodalPoints(c))  { }
Scalar EqualOrderElement::PressureValue (int q, int i) const {
    return Value_1(q,i); 
}
Gradient EqualOrderElement::PressureGradient (int q, int i) const { 
    return Derivative_1(q,i); 
}
Gradient EqualOrderElement::VelocityField (int q, int i, int k) const {
    Velocity V = 0.0;
    V[k] = Value_2(q,i);
    return V;
}
VelocityGradient EqualOrderElement::VelocityFieldGradient(int q, int i, int k) const {
    return VelocityGradient( Derivative_2(q,i),k);
}
Scalar EqualOrderElement::PressureValue (int q, const Vector& u) const { 
    Scalar p = 0.0;
    for (int i=0; i<Size_1(); ++i) p += u(r(i),Dim()) * Value_1(q,i);
    return p;
}
Gradient EqualOrderElement::PressureGradient (int q, const Vector& u) const { 
    Gradient Dp = 0.0;
    for (int i=0; i<Size_1(); ++i) Dp += u(r(i),Dim()) * Derivative_1(q,i);
    return Dp;
}
Velocity EqualOrderElement::VelocityField (int q, const Vector& u) const { 
    Velocity V = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) 
	    V[k] += u(r(i),k) * Value_2(q,i);
    return V;
}
VelocityGradient EqualOrderElement::VelocityFieldGradient (int q, const Vector& u) const { 
    VelocityGradient DV = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) {
	    Gradient G = u(r(i),k) * Derivative_2(q,i);
	    for (int l=0; l<Dim(); ++l)
		DV[k][l] += G[l];
	}
    return DV;
}
Scalar EqualOrderElement::Divergence (int q, const Vector& u) const {
    Scalar s=0.0;
    for (int i=0; i<Size_2(); ++i)
	for (int k=0; k<Dim(); ++k)
	    s += u(r(i),k) * Derivative_2(q,i)[k];
    return s;
}
Scalar EqualOrderElement::Divergence (int q, int i, int k) const {
    return Derivative_2(q,i)[k];
}


/***************************************************************************
 * EqualOrderElementPPP (P1P1 or P2P2)
***************************************************************************/
EqualOrderElementPPP::EqualOrderElementPPP (const Discretization& D, const Vector& u, 
				      const cell& c) 
    : MixedElementPPP(D,u,c,D.NodalPoints(c),D.NodalPoints(c),D.NodalPoints(c))  { }
Scalar EqualOrderElementPPP::PressureValue (int q, int i) const {
    return Value_1(q,i); 
}
Gradient EqualOrderElementPPP::PressureGradient (int q, int i) const { 
    return Derivative_1(q,i); 
}
Gradient EqualOrderElementPPP::VelocityField (int q, int i, int k) const {
    Velocity V = 0.0;
    V[k] = Value_2(q,i);
    return V;
}
VelocityGradient EqualOrderElementPPP::VelocityFieldGradient(int q, int i, int k) const {
    return VelocityGradient( Derivative_2(q,i),k);
}
Scalar EqualOrderElementPPP::PressureValue (int q, const Vector& u) const { 
    Scalar p = 0.0;
    for (int i=0; i<Size_1(); ++i) p += u(r(i),Dim()) * Value_1(q,i);
    return p;
}
Gradient EqualOrderElementPPP::PressureGradient (int q, const Vector& u) const { 
    Gradient Dp = 0.0;
    for (int i=0; i<Size_1(); ++i) Dp += u(r(i),Dim()) * Derivative_1(q,i);
    return Dp;
}
Velocity EqualOrderElementPPP::VelocityField (int q, const Vector& u) const { 
    Velocity V = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) 
	    V[k] += u(r(i),k) * Value_2(q,i);
    return V;
}
VelocityGradient EqualOrderElementPPP::VelocityFieldGradient (int q, const Vector& u) const { 
    VelocityGradient DV = 0.0;
    for (int i=0; i<Size_2(); ++i) 
	for (int k=0; k<Dim(); ++k) {
	    Gradient G = u(r(i),k) * Derivative_2(q,i);
	    for (int l=0; l<Dim(); ++l)
		DV[k][l] += G[l];
	}
    return DV;
}
Scalar EqualOrderElementPPP::Divergence (int q, const Vector& u) const {
    Scalar s=0.0;
    for (int i=0; i<Size_2(); ++i)
	for (int k=0; k<Dim(); ++k)
	    s += u(r(i),k) * Derivative_2(q,i)[k];
    return s;
}
Scalar EqualOrderElementPPP::Divergence (int q, int i, int k) const {
    return Derivative_2(q,i)[k];
}


/************************************************************************
 * RT0_P0
 * Field 1 : Pressure
 * Field 2 : Velocity
*************************************************************************/
RT0_P0Element::RT0_P0Element(const Discretization& _D, const matrixgraph& _g, const cell& _c)
    : Element(_g,_c,_D.GetQuad(_c)), 
      c(_c), dim(c.dim()),
      S_p(_D.GetShape(c,0)), S_v(_D.GetShape(c,1)),
//      value(S_p.values()), 
      size_p(1), size_v(c.Faces()) {
    for (int face=0; face<size_v; ++face) {
	Transformation T = c.GetTransformation( (*this)[face]() );
        normal[face] =  T * c.LocalFaceNormal(face);
	double dummy = norm(normal[face]);
//	mout << "dummy " <<dummy<<endl;
	normal[face] = (1.0/dummy)*normal[face];
	if (  ((*this)[face]() + normal[face]) > (*this)[face]() )
	    sign[face] = 1;
	else sign[face] = -1;
	vectorfield_face[face] = sign[face]/T.Det()
	    *T.ApplyJ(S_v.LocalVector(c.LocalFace(face),face));
	
    }
    for (int q=0; q<nQ(); ++q) {
        value[q][0] = 1.0;
        for (int i=0; i<size_v; ++i) {
//	    vectorfield[q][i] = (1.0/(GetTransformationDet(q)))
//	    vectorfield[q][i] = ( (S_v.LocalVector(LocalQPoint(q),i))
//				  *GetTransformation(q) );
	    vectorfield[q][i] = sign[i]*(1.0/GetTransformationDet(q))
                * (GetTransformation(q).ApplyJ(S_v.LocalVector(LocalQPoint(q),i)));
//                * (ApplyJacobian(q,S_v.LocalVector(LocalQPoint(q),i)));
	    divvalue[q][i] = sign[i]*(1.0/GetTransformationDet(q))
		*S_v.LocalDiv(LocalQPoint(q),i);
	}
    }
/*    mout << "=== RT0_P0 Element ========================="<<endl;
	for (int i=0; i<size_v; ++i) {
	    mout << "Point: "<<(*this)[i]() <<"     "
		 << "Normal: "<<normal[i]<<   "     "
		 << "Sign: " <<sign[i]<<      "     "
		 << "vecfield_face : "<<vectorfield_face[i]<<endl;
    }
*/
}
Velocity RT0_P0Element::VelocityField_Face (int i) const {
    return vectorfield_face[i];
} 

Scalar RT0_P0Element::PressureValue (int q, int i) const {
    return value[q][i]; 
}
Scalar RT0_P0Element::PressureValue (int q, const Vector& u) const {
    Scalar p = 0.0;
    for (int i=size_v; i<size(); ++i) p += u(r(i),0) 
					  * value[q][i-size_v];
    return p;
}
VectorField RT0_P0Element::VelocityField (int q, int i, int k) const {
    return vectorfield[q][i];
}
VectorField RT0_P0Element::VelocityField (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=0; i<size_v; ++i) V += u(r(i),k) * vectorfield[q][i];
    return V;
}
Scalar RT0_P0Element::VelocityDiv(int q, int i, int k) const {
    return divvalue[q][i];
}
Scalar RT0_P0Element::VelocityDiv(int q, const Vector& u, int k) const {
    Scalar p=0.0;
    for (int i=0; i<size_v; ++i) p += u(r(i),k) * divvalue[q][i];
    return p;
}


/***************************************************************************
 * RT0_P1
 * Field 1 : Pressure
 * Field 2 : Velocity
***************************************************************************/

RT0_P1Element::RT0_P1Element(const Discretization& _D, const matrixgraph& _g, const cell& _c)
    : Element(_g,_c,_D.GetQuad(_c)), 
      c(_c), dim(c.dim()),
      S_p(_D.GetShape(c,0)), S_v(_D.GetShape(c,1)),
      value(S_p.values()), 
      size_p(1), size_v(c.Faces()) {
    for (int q=0; q<nQ(); ++q) {
        for (int i=0; i<size_p; ++i) 
            gradient[q][i] = GetTransformation(q) * S_p.LocalGradient(q,i);
        for (int i=0; i<size_v; ++i) {
//	    vectorfield[q][i] = (1.0/(GetTransformationDet(q)))
//	    vectorfield[q][i] = ( (S_v.LocalVector(LocalQPoint(q),i))
//				  *GetTransformation(q) );
	    vectorfield[q][i] = (1.0/GetTransformationDet(q))
                * (GetTransformation(q).ApplyJ(S_v.LocalVector(LocalQPoint(q),i)));
//                * (ApplyJacobian(q,S_v.LocalVector(LocalQPoint(q),i)));
	}
    }
}

Scalar RT0_P1Element::PressureValue (int q, int i) const {
    return value[q][i]; 
}
Gradient RT0_P1Element::PressureGradient (int q, int i) const {
    return gradient[q][i];
}
Scalar RT0_P1Element::PressureValue (int q, const Vector& u) const {
    Scalar p = 0.0;
    for (int i=0; i<size_p; ++i) p += u(r(i),0) * value[q][i];
    return p;
}
Gradient RT0_P1Element::PressureGradient (int q, const Vector& u) const {
    Gradient Dp = 0.0;
    for (int i=0; i<size_p; ++i) Dp += u(r(i),0) * gradient[q][i];
    return Dp;
}
VectorField RT0_P1Element::VelocityField (int q, int i, int k) const {
    return vectorfield[q][i];
}
VectorField RT0_P1Element::VelocityField (int q, const Vector& u, int k) const {
    VectorField V = 0.0;
    for (int i=size_p; i<size(); ++i) V += u(r(i),k) * vectorfield[q][i];
    return V;
}
