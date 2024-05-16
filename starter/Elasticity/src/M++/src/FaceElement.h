// file:   FaceElement.h
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/FaceElement.h,v 1.12 2009-01-09 17:57:53 mueller Exp $

#ifndef _FACEELEMENT_H_
#define _FACEELEMENT_H_

#include "Cell.h"
#include "Discretization.h"
#include "Quadrature.h" 
#include "Algebra.h" 

class FaceElement : public rows {
    const Quadrature& Q;
    Point qLocal[MaxQuadraturePoints];
    Point qPoint[MaxQuadraturePoints];
    Point qNormal[MaxQuadraturePoints];
    double qWeight[MaxQuadraturePoints];
    Transformation T[MaxQuadraturePoints];
 protected:
    const row& r (int i) const { return (*this)[i]; }
 public:
    FaceElement (const matrixgraph& g, const cell& c, int face, 
		 const Quadrature& Quad);
    
    int nQ () const { return Q.size(); }
    const Point& NodalPoint (int i) const  { return (*this)[i](); }
    const Point& LocalQPoint (int q) const { return Q.QPoint(q); }
    const Point& QLocal (int q) const      { return qLocal[q]; }
    const Point& QPoint (int q) const      { return qPoint[q]; }
    const Point& QNormal (int q) const     { return qNormal[q]; }
    double QWeight (int q) const { return qWeight[q]; }
    const Transformation& GetTransformation (int q) const { return T[q]; }
    double GetTransformationDet (int q) const { return T[q].Det(); }
    double Area () const;
};


/* nuetzlich waere ein weiteres Argument "int p" im konstruktor
   um im Konstruktor "Shape" abhaengig vom Polynomgrad p zu 
   initialisieren, d.h. S(D.GetShape(c, p )
*/
class ScalarFaceElement : public FaceElement {
 protected:
    const Shape& S;
    ShapeValue value;
    ShapeGradient gradient;
 public:
    ScalarFaceElement (const Discretization& D, const matrixgraph& g, 
		       const cell& c, int face, int p = 0);
    virtual Scalar Value (int q, int i) const { return value[q][i]; }
    virtual Scalar Value (int q, const Vector& u, int k=0) const;
    Gradient Derivative (int q, int i) const { return gradient[q][i]; }
    friend ostream& operator << (ostream& s, const ScalarFaceElement& E);
};

class CurlFaceElement : public FaceElement {
 protected:
    const Shape& S;
    ShapeGradient vectorfield;
    ShapeGradient curlfield;
 public:
    CurlFaceElement (const Discretization& D, const matrixgraph& g, 
		     const cell& c, int face, int p = 0);
    VectorField VectorValue (int q, int i) const { return vectorfield[q][i]; }
    VectorField CurlVector (int q, int i) const { return curlfield[q][i]; }
    friend ostream& operator << (ostream& s, const CurlFaceElement& E);
};

class VectorFieldFaceElement : public ScalarFaceElement {
 protected:
    int dim;
 public:
    VectorFieldFaceElement (const Discretization& D, const matrixgraph& g, 
			    const cell& c, int face, int p = 0) 
	: ScalarFaceElement(D,g,c,face,p), dim(c.dim()) { }

    virtual VectorField VectorValue (int q, int i, int k) const;
    
    virtual VectorField VectorValue (int q, const Vector& u) const;
    
//    Tensor VectorGradient (int q, int i, int k) const {
//	Tensor T = 0.0;
//	T[k] = Derivative(q,i);
//	return T;
//    }
};

class MixedFaceElement : public FaceElement {
 protected:
    const Shape& S1;
    ShapeValue value1;
    ShapeGradient gradient1;
    const Shape& S2;    
    ShapeValue value2;
    ShapeGradient gradient2;
    int dim;
    int size_1;
    int size_2;
    const cell& C;
 public:
    MixedFaceElement (const Discretization& D, const matrixgraph& g, 
		      const cell& c, int face);
    int Size_1() const { return size_1; }
    int Size_2() const { return size_2; }
    int Dim() const { return dim; }
    Scalar Value_2 (int q, int i) const { return value2[q][i]; }
    Scalar Value_2 (const Point& z, const Vector& u, int k=0) const;
    Gradient Derivative_1 (int q,int i) const { return gradient1[q][i]; }
    Gradient Derivative_1 (const Point& z, const Vector& u, int k=0) const;
    friend ostream& operator << (ostream& s, const MixedFaceElement& E);
};

class RT0FaceElement : public FaceElement {
 protected:   
    const Shape& S;
    Point vectorfield[MaxQuadraturePoints];
//    ShapeGradient vectorfield;
    int sign;   
    Point normal;
 public:
    RT0FaceElement (const Discretization& D, const matrixgraph& g,
		    const cell& c, int face, int p = 0);
    virtual VectorField VectorValue (int q, int k=0) const;
    virtual VectorField VectorValue (int q, const Vector& u, int k=0) const;
    int Sign() const { return sign; }
    Point Normal() const { return normal; }
};

#endif
