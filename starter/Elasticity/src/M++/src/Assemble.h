// file:   Assemble.h
// author: Christian Wieners
// $Header: /public/M++/src/Assemble.h,v 1.37 2009-03-06 11:22:12 mueller Exp $

#ifndef _ASSEMBLE_H_
#define _ASSEMBLE_H_

#include "Algebra.h"

const int FREE_BC             =   0;
const int DIRICHLET_BC        =   1; //Dirichlet b.c. for x, y, and z-direction
const int NEUMANN_BC          =   2;
const int CONTACT_BC          =   7;
const int DIRICHLET_X_BC      = 100; //Dirichlet b.c. for x-direction
const int DIRICHLET_Y_BC      = 101; //Dirichlet b.c. for y-direction
const int DIRICHLET_Z_BC      = 102; //Dirichlet b.c. for z-direction
const int DIRICHLET_SYM_X_BC  = 110; //Dirichlet symmetry b.c. for x-dir (u_x=0)
const int DIRICHLET_SYM_Y_BC  = 111; //Dirichlet symmetry b.c. for y-dir (u_y=0)
const int DIRICHLET_SYM_Z_BC  = 112; //Dirichlet symmetry b.c. for z-dir (u_z=0)
const int DIRICHLET_FREE_X_BC = 131; //Dirichlet b.c. for y- and z-direction
const int DIRICHLET_FREE_Y_BC = 132; //Dirichlet b.c. for x- and z-direction
const int DIRICHLET_FREE_Z_BC = 133; //Dirichlet b.c. for x- and y-direction
const int DIRICHLET_FIX_BC    = 199; //Dirichlet b.c. all dir (u_x= u_y= u_z =0)
const int HOLE_BC             = 333; //This is a hole

class Assemble {
 protected:
    bool symmetric;
    int verbose;

    void Dirichlet (int dim, 
		    int k, RowBndValues& u_c, const VectorField& D) const {
	for (int d=0; d<dim; ++d) {
	    u_c(k,d)   = D[d];
	    u_c.D(k,d) = true;
	}
    }
    void Dirichlet_X (int k, RowBndValues& u_c, const VectorField& D) const {
	u_c(k,0)   = D[0];		      			 
	u_c.D(k,0) = true;		      			 
    }
    void Dirichlet_Y (int k, RowBndValues& u_c, const VectorField& D) const {
	u_c(k,1)   = D[1];		      			 
	u_c.D(k,1) = true;		      			 
    }
    void Dirichlet_Z (int k, RowBndValues& u_c, const VectorField& D) const {
	u_c(k,2)   = D[2];
	u_c.D(k,2) = true;
    }
    void Dirichlet_FIX (int dim, int k, RowBndValues& u_c) const{
	for (int d=0; d<dim; ++d) {
	    u_c(k,d)   = 0.0;
	    u_c.D(k,d) = true;
	}
    }   
    void Dirichlet_SYM_X (int k, RowBndValues& u_c) const {
	u_c(k,0)   = 0.0;
	u_c.D(k,0) = true;
    }
    void Dirichlet_SYM_Y (int k, RowBndValues& u_c) const {
	u_c(k,1)   = 0.0;
	u_c.D(k,1) = true;
    }
    void Dirichlet_SYM_Z (int k, RowBndValues& u_c) const {
	u_c(k,2)   = 0.0;
	u_c.D(k,2) = true;
    }
    bool SetDirichletValues (int bc, int dim, int k, 
			     RowBndValues& u_c, const VectorField& D) const {
	switch (bc) {
	    case FREE_BC:                                        return true;
	    case DIRICHLET_BC:        Dirichlet(dim,k,u_c,D);    return true;
	    case DIRICHLET_X_BC:      Dirichlet_X(k,u_c,D);      return true;
	    case DIRICHLET_Y_BC:      Dirichlet_Y(k,u_c,D);      return true;
	    case DIRICHLET_Z_BC:      Dirichlet_Z(k,u_c,D);      return true;
	    case DIRICHLET_FIX_BC:    Dirichlet_FIX(dim,k,u_c);  return true;
	    case DIRICHLET_SYM_X_BC:  Dirichlet_SYM_X(k,u_c);    return true;
	    case DIRICHLET_SYM_Y_BC:  Dirichlet_SYM_Y(k,u_c);    return true;
	    case DIRICHLET_SYM_Z_BC:  Dirichlet_SYM_Z(k,u_c);    return true;
	}
	return false;
    }
    bool Dirichlet_PrescribedDisplacement_BC (int bc) const {
	switch (bc) {
	    case DIRICHLET_BC:           return true;
	    case DIRICHLET_X_BC:         return true;
	    case DIRICHLET_Y_BC:         return true;
	    case DIRICHLET_Z_BC:         return true;
	}
	return false;
    }
    bool Dirichlet_Fixed_BC (int bc) const {
	switch (bc) {
	    case DIRICHLET_FIX_BC:       return true;
	    case DIRICHLET_SYM_X_BC:     return true;
	    case DIRICHLET_SYM_Y_BC:     return true;
	    case DIRICHLET_SYM_Z_BC:     return true;
	}
	return false;
    }
 public:
    Assemble (bool sym = false) : symmetric(sym), verbose(1) {
	ReadConfig(Settings,"AssembleVerbose",verbose);
    }
    virtual void Dirichlet (Vector&) const {}
    virtual double Energy (const Vector&) const {}
    virtual double Residual (const Vector&, Vector&) const = 0;
    virtual void Jacobi (const Vector&, Matrix&) const  = 0;
    virtual void UpdateSQP (const Vector&) {} 
    virtual double KKT (const Vector&, double&, double& ) {} 
    virtual void Update( Vector&, const Vector&) const {}
};

class TAssemble : public Assemble {
 protected:
    const    Vector *u_;
    const    Vector *v_;  // Zeigen auf xxx_old
    const    Vector *a_;
 public:
    const Vector& U_old () const { return *u_;  }
    const Vector& V_old () const { return *v_; }
    const Vector& A_old () const { return *a_; }
 protected:
    int step;
    double beta;
    double gamma;
    double t_;
    double dt_;
 protected:
    void Dirichlet (Vector& u) const { Dirichlet(t_,u); }
    double Residual (const Vector& u, Vector& b) const { 
	return Residual(t_,u,b); 
    }
    void Jacobi (const Vector& u, Matrix& A) const { 
	Date Start;
	Jacobi(t_,u,A); 
	tout(1) << "   assemble time " << Date() - Start << endl;
    } 
 public:
    TAssemble (bool sym = false) : Assemble(sym), step(0), t_(0) { }
    virtual ~TAssemble () {}
    virtual void Initialize (double t) { 
	t_ = t; 
    }
    virtual void Initialize (double t, const Vector& u) {
	t_ = t;
    }
    virtual void Initialize (double t, const Vector& u, const Vector& v, Vector& a) { 
	t_ = t; 
    }
    virtual void BeforeInitTimeStep (double t_old, double t, double t_new,
				     const Vector& u_old,
				     const Vector& u,
				     Vector& u_new) {}
    virtual void InitTimeStep (double t_old, double t, double t_new,
			       const Vector& u_old,
			       const Vector& u,
			       Vector& u_new) {
	BeforeInitTimeStep(t_old,t,t_new,u_old,u,u_new);
	t_  = t_new;  
   	dt_ = t_new - t;
	u_ = &u_old;
    }
    virtual void InitTimeStep (double t_old, double t, double t_new,
			       const Vector& u_old,
			       const Vector& u,
			       const Vector& u_new,
			       const Vector& v_old,
			       const Vector& a_old) {
	t_  = t_new;  
   	dt_ = t_new - t;
	u_ = &u_old;  
	v_ = &v_old;
	a_ = &a_old;
    }
    virtual void BeforeFinishTimeStep (double t, const Vector& u_new, 
				       bool SpecialTimeStep = false) { }
    virtual void AfterFinishTimeStep (double t, const Vector& u_new, 
				      bool SpecialTimeStep = false) { }
    virtual void FinishTimeStep (double t, const Vector& u_new, 
				 bool SpecialTimeStep = false) { 
	BeforeFinishTimeStep(t,u_new,SpecialTimeStep);
	++step; 
	AfterFinishTimeStep(t,u_new,SpecialTimeStep);
    }
    virtual void FinishTimeStep (double t, const Vector& u_new, 
				 const Vector& v_new,
				 bool SpecialTimeStep = false) { 
	++step; 
    }
    virtual void Finalize (double t, const Vector& u_new) {}
    virtual void Finalize (double t, const Vector& u_new,
			   const Vector& v_new,
			   const Vector& a_new) { }
    virtual bool BlowUp (const Vector& u_new) const { return false; }
    virtual void Dirichlet (double t, const cell& c, Vector& u) const {}
    virtual void Dirichlet (double t, Vector& u) const {
	for (cell c=u.cells(); c!=u.cells_end(); ++c) Dirichlet(t,c,u);
	DirichletConsistent(u);
    }
    virtual void Residual (double t, const cell& c, const Vector& u, 
			   Vector& b) const {} 
    virtual double Residual (double t, const Vector& u, Vector& b) const {
	b = 0;
	for (cell c = u.cells(); c != u.cells_end(); ++c) 
	    Residual(t,c,u,b);
	b.ClearDirichletValues () ; 
	Collect(b);
	return b.norm();
    }
    virtual void Jacobi (double t, const cell& c, const Vector& u, 
			 Matrix& A) const {}
    virtual void Jacobi (double t, const Vector& u, Matrix& A) const {
	A = 0;
	for (cell c = u.cells(); c != u.cells_end(); ++c) Jacobi(t,c,u,A);
	if (symmetric) A.Symmetric();
	A.ClearDirichletValues();
    }
    virtual void MassMatrix(const Vector& u, Matrix& A) const { }
    virtual void RHS(double t,const Vector& u, Vector& b) const {}
    virtual double Energy (double t, const cell& c, const Vector& u) const { 
	return 0;
    }
    virtual double Energy (double t, const Vector& u) const {
	double E = 0;
	for (cell c = u.cells(); c != u.cells_end(); ++c) {
	    E += Energy(t,c,u);
	}
	return PPM->Sum(E);
    }
    virtual double Energy (const Vector& u) const {
	return Energy(t_,u); 	
    }
    int Step () const { return step; }
    void SetNewmarkParameters (double _gamma, double _beta) {
	beta = _beta;
	gamma = _gamma;
    }
};

class HAssemble : public Assemble {
    double t_;
    int step;
    mutable int level;
private:
    virtual int find_level (const Vector& u) const {}
    void read_level (const Vector& u) const {
        level = find_level(u);
    }
public:
    HAssemble (bool sym = false) : Assemble(sym), step(0), t_(0) { }
    virtual ~HAssemble () {}
    int Level () const { return level; }
    void Dirichlet (Vector& u) const {
        read_level(u);
        Dirichlet(t_,u);
    }
    double Energy (Vector& u) const {
        read_level(u);
        return Energy(t_,u);
    }
    double Residual (const Vector& u, Vector& b) const {
        read_level(u);
        return Residual(t_,u,b);
    }
    void Jacobi (const Vector& u, Matrix& A) const {
        Date Start;
        read_level(u);        
	Jacobi(t_,u,A); 
	tout(1) << "   assemble time " << Date() - Start << endl;
    }
    void AfterNewtonDefect (const Vector& u) const {
        CountPlasticPoints(u);
    }    
    virtual void NewtonMinusCorrection(Vector &u, const Vector& c) {}
    virtual void NewtonPlusCorrection(Vector &u, const Vector& c) {}
    virtual void NewtonUpdate (Vector& u, const Vector& c) {}
    virtual void NewtonBackdate (Vector& u, const Vector& c) {}
    virtual void ExtrapolateRotationVector (double s) {}
    virtual void ExtrapolateRotationVector (double s, const Vector& u) {}
    virtual void ResetRotationVector () {}    
    virtual void ResetRotationVector (const Vector& u) {}    
    virtual void project_eulerian_angle (Vector& c) {}
private:
    virtual void Dirichlet (double t, const cell& c, Vector& u) const {}
    virtual void Dirichlet (double t, Vector& u) const {
	for (cell c=u.cells(); c!=u.cells_end(); ++c)
            Dirichlet(t,c,u);
	DirichletConsistent(u);
    }
    virtual bool detailed_energies () const { return false; }
    virtual void reset_energies () const {}
    virtual double sum_energies () const { return 0; }
    virtual double Energy (double t, const cell& c, Vector& u) const {}
    virtual double Energy (double t, Vector& u) const {
        if (detailed_energies()) {
            reset_energies();
            for (cell c=u.cells(); c!=u.cells_end(); ++c)
                Energy(t,c,u);
            return sum_energies();
        }
        else {
            double E = 0;
            for (cell c=u.cells(); c!=u.cells_end(); ++c)
                E += Energy(t,c,u);
            E = PPM->Sum(E);
            return E;
        }
    }
    virtual void Residual (double t, const cell& c, const Vector& u,
                           Vector& b) const = 0;
    virtual double Residual (double t, const Vector& u, Vector& b) const {
	b = 0;
	for (cell c = u.cells(); c != u.cells_end(); ++c) 
	    Residual(t,c,u,b);
	b.ClearDirichletValues () ; 
	Collect(b);
	return b.norm();
    }
    virtual void Jacobi (double t, const cell& c, const Vector& u, 
			 Matrix& A) const = 0;
    virtual void Jacobi (double t, const Vector& u, Matrix& A) const {
	A = 0;
	for (cell c = u.cells(); c != u.cells_end(); ++c) Jacobi(t,c,u,A);
	if (symmetric) A.Symmetric();
	A.ClearDirichletValues();
    }
public:
    void Initialize (double t, Vector& u) {
        Init(t,u);
        t_ = t;
    }
    void InitTimeStep (double t_old, double t, double t_new,
                       const Vector& u_old, const Vector& u,
                       Vector& u_new) {
        InitTStep(t_old,t,t_new,u_old,u,u_new);
	t_  = t_new;  
    }
    void Finalize (double t_new, const Vector& u_new) {
        Finish(t_new,u_new);
    }
    void FinishTimeStep (double t_new, const Vector& u_new, 
                         bool SpecialTimeStep = true) {
        FinishTStep(t_new,u_new,SpecialTimeStep);
	++step; 
    }
    virtual void CountPlasticPoints (const Vector& u) const {}
private:
    virtual void Init (double t, Vector& u) {}
    virtual void InitTStep (double t_old, double t, double t_new,
                           const Vector& u_old, const Vector& u,
                           Vector& u_new) {}
    virtual void Finish (double t_new, const Vector& u_new) {}
    virtual void FinishTStep (double t_new, const Vector& u_new, bool SpecialTimeStep = true) {}
public:
    virtual bool BlowUp (const Vector& u_new) const { return false; }
    int Step () const { return step; }
};

#endif
