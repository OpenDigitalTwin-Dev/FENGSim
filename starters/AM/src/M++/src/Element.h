// file: Element.h
// author: Christian Wieners
// $Header: /public/M++/src/Element.h,v 1.15 2009-10-21 09:58:56 wieners Exp $

#ifndef _ELEMENT_H_
#define _ELEMENT_H_

#include "Discretization.h"
#include "Algebra.h"

class Element: public rows {
    const Quadrature& Q;
    double qWeight[MaxQuadraturePoints];
    Point qPoint[MaxQuadraturePoints];
    Transformation T[MaxQuadraturePoints];
 protected:
    const row& r (int i) const { return (*this)[i]; }
 public:
    Element (const matrixgraph& g, const cell& c, const Quadrature& Quad);
    int nQ () const { return Q.size(); }
    double QWeight (int q) const { return qWeight[q]; }
    const Point& NodalPoint (int i) const { return (*this)[i](); }
    const Point& LocalQPoint (int q) const { return Q.QPoint(q); }
    const Point& QPoint (int q) const { return qPoint[q]; }
    const Transformation& GetTransformation (int q) const { return T[q]; }
    Point ApplyJacobian (int q, const Point& x) const { return T[q](x); }
    double GetTransformationDet (int q) const { return T[q].Det(); }
    double Area () const { 
	double a = 0;
	for (int q=0; q<nQ(); ++q)
	    a +=qWeight[q]; 
	return a;
    }
};

class ScalarElement: public Element {
    const Shape& S;
    const ShapeValue& value;
    ShapeGradient gradient;
    const cell& C;
 public:
    ScalarElement(const Discretization& D,const matrixgraph& g,const cell& c);
    Scalar Value (int q, int i) const { return value[q][i]; }
    Scalar Value (int q, const Vector& u, int k=0) const;
    
    Scalar Value (const Point& z, const Vector& u, int k=0) const;

    Gradient Derivative (int q, int i) const { return gradient[q][i]; }
    Gradient Derivative (int q, const Vector& u, int k=0) const;
    
    Gradient Derivative (const Point& z, const Vector& u, int k=0) const;
    
    friend ostream& operator << (ostream& s, const ScalarElement& E);


    const cell& _cell () { return C; } 
};

class VectorFieldElement: public ScalarElement {
    int dim;
public:
    VectorFieldElement (const Discretization& D, const matrixgraph& g,
						const cell& c);
    VectorField VectorValue (int q, int i, int k) const;
    
    VectorField VectorValue (int q, const Vector& u) const;
	
    VectorField VectorValue (const Point& L,  const Vector& u) const;
    
    Tensor VectorGradient (int q, int i, int k) const;
    
    Tensor VectorGradient (int q,  const Vector& u) const;
    
    Tensor VectorGradient (const Point& L,  const Vector& u) const;
	
    Scalar Divergence (int q, int i, int k) const;
    
    Scalar Divergence (int q, const Vector& u) const;
};

class CurlElement: public Element {
    const Shape& S;
    const cell& c;
    double sign[MaxNodalPoints];
    Point tangent[MaxNodalPoints];
    ShapeGradient vectorfield;
    ShapeGradient curlfield;

    Point DofShape_int(double x, double y, double z, int i) const;
 public:
    CurlElement (const Discretization& D,const matrixgraph& g,const cell& _c);
    
    VectorField VectorValue (int q, int i) const { return vectorfield[q][i]; }
    VectorField CurlVector (int q, int i) const { return curlfield[q][i]; }
    VectorField VectorValue (int q, const Vector& u, int k=0) const;
    VectorField VectorValue (const Point&, const Vector&, int k=0) const;
    VectorField CurlVector (int q, const Vector& u, int k=0) const;
    VectorField CurlVector (const Point&, const Vector&, int k=0) const;
    Point DofShape(const Point& z, int i) const { return DofShape_int(z[0],z[1],z[2],i); }
    Point TangentVector (int i) const { return tangent[i]; }
//    short FaceNodalPoints (int i) const { return c.FaceEdges(i); }
//    short FaceNodalPoint (int i, int j) const { return c.faceedge(i,j); }
    friend ostream& operator << (ostream& s, const CurlElement& E);
    double GetSign (int i) const { return sign[i]; }
};

class QuasiMagnetoStaticsMixedElement : public Element {
 protected:
    const Shape& S1;
    const ShapeValue& value;
    ShapeGradient gradient;
    const Shape& S2;    
    double sign[MaxNodalPoints];
    Point tangent[MaxNodalPoints];
    ShapeGradient vectorfield;
    ShapeGradient curlfield;
    int dim;
    int size_1;
    int size_2;
    const cell& C;
 public:
    QuasiMagnetoStaticsMixedElement (const Discretization& D, const matrixgraph& g, const cell& c);
    int Size_1() const { return size_1; }
    int Size_2() const { return size_2; }
    int Dim() const { return dim; }
    //  scalar element
    Scalar Value (int q, int i) const { return value[q][i]; }
    Scalar Value (int q, const Vector& u, int k=0) const;
    Gradient Derivative (int q,int i) const { return gradient[q][i]; }
    Gradient Derivative (int q, const Vector& u, int k=0) const;
    Gradient Derivative (const Point& z, const Vector& u, int k=0) const;
    //  vector element
    VectorField VectorValue (int q, int i) const { return vectorfield[q][i]; }
    VectorField VectorValue (int q, const Vector& u, int k=0) const;
    VectorField VectorValue (const Point&, const Vector&, int k=0) const;
    VectorField CurlVector (int q, int i) const { return curlfield[q][i]; }
    VectorField CurlVector (int q, const Vector& u, int k=0) const;
    VectorField CurlVector (const Point&, const Vector&, int k=0) const;

    friend ostream& operator << (ostream& s, const QuasiMagnetoStaticsMixedElement& E);
};

class MixedElement : public Element {
 protected:
    const Shape& S1;
    const ShapeValue& value1;
    ShapeGradient gradient1;
    const Shape& S2;    
    const ShapeValue& value2;
    ShapeGradient gradient2;
    int dim;
    int size_1;
    int size_2;
    const cell& C;
 public:
    MixedElement (const Discretization& D, const Vector& u, const cell& c,
		  int _size_1, int _size_2); 
    MixedElement (const Discretization& D, const matrixgraph& g, const cell& c);
    int Size_1() const { return size_1; }
    int Size_2() const { return size_2; }
    int Dim() const { return dim; }
    Scalar Value_1 (int q, int i) const { return value1[q][i]; }
    Scalar Value_2 (int q, int i) const { return value2[q][i]; }
    Scalar Value_1 (const Point& z, const Vector& u, int k=0) const;
    Scalar Value_2 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_1 (int q,int i) const { return gradient1[q][i]; }
    Gradient Derivative_2 (int q,int i) const { return gradient2[q][i]; }
    Gradient Derivative_1 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_2 (const Point& z, const Vector& u, int k=0) const;
    friend ostream& operator << (ostream& s, const MixedElement& E);
};

class MixedElementPPP : public Element {
 protected:
    const Shape& S1;
    const ShapeValue& value1;
    ShapeGradient gradient1;
    const Shape& S2;    
    const ShapeValue& value2;
    ShapeGradient gradient2;
    const Shape& S3;    
    const ShapeValue& value3;
    ShapeGradient gradient3;
    int dim;
    int size_1;
    int size_2;
    int size_3;
    const cell& C;
 public:
    MixedElementPPP (const Discretization& D, const Vector& u, const cell& c,
		  int _size_1, int _size_2, int _size_3); 
    MixedElementPPP (const Discretization& D, const matrixgraph& g, const cell& c);
    int Size_1() const { return size_1; }
    int Size_2() const { return size_2; }
    int Size_3() const { return size_3; }
    int Dim() const { return dim; }
    Scalar Value_1 (int q, int i) const { return value1[q][i]; }
    Scalar Value_2 (int q, int i) const { return value2[q][i]; }
    Scalar Value_3 (int q, int i) const { return value3[q][i]; }
    Scalar Value_1 (const Point& z, const Vector& u, int k=0) const;
    Scalar Value_2 (const Point& z, const Vector& u, int k=0) const;
    Scalar Value_3 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_1 (int q,int i) const { return gradient1[q][i]; }
    Gradient Derivative_2 (int q,int i) const { return gradient2[q][i]; }
    Gradient Derivative_3 (int q,int i) const { return gradient3[q][i]; }
    Gradient Derivative_1 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_2 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_3 (const Point& z, const Vector& u, int k=0) const;
    friend ostream& operator << (ostream& s, const MixedElementPPP& E);
};

/***************************************************************************
 * TaylorHoodElement
 * Field 1 : Pressure
 * Field 2 : Velocity / Displacement
 ***************************************************************************/

class TaylorHoodElement : public MixedElement {

 public:
    TaylorHoodElement (const Discretization& D, const Vector& u, 
		       const cell& c);
    Scalar PressureValue (int q, int i) const; 
    Gradient PressureGradient (int q, int i) const;
    Gradient VelocityField (int q, int i, int k) const;
    VelocityGradient VelocityFieldGradient (int q, int i, int k) const;
    Scalar PressureValue (int q, const Vector& u) const;
    Gradient PressureGradient (int q, const Vector& u) const;
    Velocity VelocityField (int q, const Vector& u) const;
    VelocityGradient VelocityFieldGradient (int q, const Vector& u) const;
    Scalar Divergence (int q, const Vector& u) const;
    Scalar Divergence (int q, int i, int k) const;
};

/***************************************************************************
 * EqualOrderElement (P1P1 or P2P2 ... (unstable for Stokes...)
 * Field 1 : Pressure
 * Field 2 : Velocity / Displacement
 ***************************************************************************/

class EqualOrderElement : public MixedElement {

 public:
    EqualOrderElement (const Discretization& D, const Vector& u, 
		       const cell& c);
    Scalar PressureValue (int q, int i) const; 
    Gradient PressureGradient (int q, int i) const;
    Gradient VelocityField (int q, int i, int k) const;
    VelocityGradient VelocityFieldGradient (int q, int i, int k) const;
    Scalar PressureValue (int q, const Vector& u) const;
    Gradient PressureGradient (int q, const Vector& u) const;
    Velocity VelocityField (int q, const Vector& u) const;
    VelocityGradient VelocityFieldGradient (int q, const Vector& u) const;
    Scalar Divergence (int q, const Vector& u) const;
    Scalar Divergence (int q, int i, int k) const;
};


/***************************************************************************
 * EqualOrderElement (P1P1 or P2P2 ... (unstable for Stokes...)
 * Field 1 : Pressure
 * Field 2 : Velocity / Displacement
 ***************************************************************************/

class EqualOrderElementPPP : public MixedElementPPP {

 public:
    EqualOrderElementPPP (const Discretization& D, const Vector& u, 
		       const cell& c);
    Scalar PressureValue (int q, int i) const; 
    Gradient PressureGradient (int q, int i) const;
    Gradient VelocityField (int q, int i, int k) const;
    VelocityGradient VelocityFieldGradient (int q, int i, int k) const;
    Scalar PressureValue (int q, const Vector& u) const;
    Gradient PressureGradient (int q, const Vector& u) const;
    Velocity VelocityField (int q, const Vector& u) const;
    VelocityGradient VelocityFieldGradient (int q, const Vector& u) const;
    Scalar Divergence (int q, const Vector& u) const;
    Scalar Divergence (int q, int i, int k) const;
};


/*
virtual class AbstractMixedElement : public Element {
 protected:
    int fields;
    int dim;
    vector<const Shape&> S;
    vector<int> fieldsize;
    const cell& c;
 public:
    virtual AbstractMixedElement (const Discretization& _D,const matrixgraph& _g,
			  const cell& _c, int _fields);
};
*/
/***************************************************************************
 * RT0_P0
 * Field 1 : Pressure
 * Field 2 : Velocity
 ***************************************************************************/

class RT0_P0Element : public Element {
    const cell& c;
    int dim;
    const Shape& S_p;
    const Shape& S_v;
    ShapeGradient vectorfield;
    Point vectorfield_face[MaxNodalPoints];
    ShapeValue value;
    ShapeValue divvalue;
    int size_p;
    int size_v;
    Point normal[MaxNodalPoints];
    double sign[MaxNodalPoints];
 public:
    RT0_P0Element(const Discretization& _D, const matrixgraph& _g, 
		  const cell& _c);
    Scalar PressureValue (int q, int i) const; 
    VectorField VelocityField (int q, int i, int k=0) const;
    Scalar PressureValue (int q, const Vector& u) const;
    Velocity VelocityField (int q, const Vector& u, int k=0) const;
    Velocity VelocityField_Face (int i) const; 
    Scalar VelocityDiv(int q, int i, int k=0) const;
    Scalar VelocityDiv(int q, const Vector& u, int k=0) const;
    int Size_p() const { return size_p; }
    int Size_v() const { return size_v; }
    Point Normal (int i) const { return normal[i]; }
    double Sign (int i) const { return sign[i]; }
};
/***************************************************************************
 * RT0_P1
 * Field 1 : Pressure
 * Field 2 : Velocity
 ***************************************************************************/

class RT0_P1Element : public Element {
    const cell& c;
    int dim;
    const Shape& S_p;
    const Shape& S_v;
    ShapeGradient vectorfield;
    const ShapeValue& value;
    ShapeGradient gradient;
    ShapeValue divvalue;
    int size_p;
    int size_v;
    Point normal[MaxNodalPoints];
    double sign[MaxNodalPoints];
 public:
    RT0_P1Element(const Discretization& _D, const matrixgraph& _g, const cell& _c);
    Scalar PressureValue (int q, int i) const; 
    Gradient PressureGradient (int q, int i) const;
    VectorField VelocityField (int q, int i, int k=0) const;
    Scalar PressureValue (int q, const Vector& u) const;
    Gradient PressureGradient (int q, const Vector& u) const;
    Velocity VelocityField (int q, const Vector& u, int k=0) const;
    Scalar VelocityDiv(int q, int i, int k=0) const;
    Scalar VelocityDiv(int q, const Vector& u, int k=0) const;
    int Size_p() const { return size_p; }
    int Size_v() const { return size_v; }
    Point Normal (int i) const { return normal[i]; }
    double Sign (int i) const { return sign[i]; }
};

#endif
