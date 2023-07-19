// =======================================================================
// created by jiping 2017.10
// data and data structure for primitives like cube, sphere, cylinder, cone
// , torus
// =======================================================================
#ifndef PRIMITIVE_H
#define PRIMITIVE_H
#include "TopoDS_Shape.hxx"
#include "BRepPrimAPI_MakeBox.hxx"
#include "BRepPrimAPI_MakeSphere.hxx"
#include "BRepPrimAPI_MakeCylinder.hxx"
#include "BRepPrimAPI_MakeCone.hxx"
#include "BRepPrimAPI_MakeTorus.hxx"
#include "vector"
//#include "ExternWidgets.h"
#include <QTableWidget>
// PrimSet is a set which includes created primitives.
// SelectedPrimSet is a set which includes selected primitives;
// PrimIsInclude is a function to check if S is in PrimSet.
class Primitive;
extern std::vector<Primitive*> PrimSet;
int PrimIsInclude (TopoDS_Shape S);
#include "CAD/CADDockWidget.h"
#include "ui_CADDockWidget.h"
#include "CAD/PhysicsDockWidget.h"
#include "ui_PhysicsDockWidget.h"
// =======================================================================
// a general class with parameters from TopoDS_Shape
// =======================================================================
enum PrimType
{
    any, cube, sphere, cylinder, cone, torus, vertex, line, plane
};
class Primitive
{
protected:
    PrimType T;
    TopoDS_Shape* S;
public:
    Primitive (TopoDS_Shape* _S) : S(_S) {}
    PrimType type ()
    {
        return T;
    }
    TopoDS_Shape* Value ()
    {
        return S;
    }
    virtual void ShowProperties (CADDockWidget*) = 0;
    QString PositionStr (double x, double y, double z)
    {
        return "("
                + QString::number(x) + ","
                + QString::number(y) + ","
                + QString::number(z) +
                ")";
    }
    virtual void SetPosition (double, double, double) {}
    virtual void GetPosition (double&, double&, double&) {}
};
// =======================================================================
// a class for any shape which is not cube, cylinder, cone, torus, sphere
// =======================================================================
class General : public Primitive
{
    double pos[3];
    double dir[3];
    double angle;
public:
    General (TopoDS_Shape* S) : Primitive(S)
    {
        T = PrimType::any;
        pos[0] = 0;
        pos[1] = 0;
        pos[2] = 0;
        dir[0] = 0;
        dir[1] = 0;
        dir[2] = 1;
        angle = 0;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetAnyModule(pos, dir, angle);
    }
    void GetPosition (double& p1, double& p2, double& p3)
    {
        p1 = pos[0];
        p2 = pos[1];
        p3 = pos[2];
    }
    void SetPosition (double p1, double p2, double p3)
    {
        pos[0] = p1;
        pos[1] = p2;
        pos[2] = p3;
    }
    void SetDirection (double d1, double d2, double d3)
    {
        dir[0] = d1;
        dir[1] = d2;
        dir[2] = d3;
    }
    void SetAngle (double an)
    {
        angle = an;
    }
};
// =======================================================================
// a class for point with parameters for x,y,z
// =======================================================================
#include "BRepBuilderAPI_MakeVertex.hxx"
class Vertex : public Primitive
{
    double x[3];
public:
    Vertex (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double x1, double x2, double x3)
    {
        x[0] = x1;
        x[1] = x2;
        x[2] = x3;
        T = PrimType::vertex;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetPointModule(x);
    }
};
// =======================================================================
// a class for a line with two vertices
// =======================================================================
#include "BRepBuilderAPI_MakeEdge.hxx"
class Line : public Primitive
{
    double x1[3];
    double x2[3];
public:
    Line (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double x11=0, double x12=0, double x13=0,
                  double x21=100, double x22=0, double x23=0)
    {
        x1[0] = x11;
        x1[1] = x12;
        x1[2] = x13;
        x2[0] = x21;
        x2[1] = x22;
        x2[2] = x23;
        T = PrimType::line;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetLineModule(x1,x2);
    }
};
// =======================================================================
// a class for a plane with two vertices
// =======================================================================
#include "BRepBuilderAPI_MakeFace.hxx"
#include "gp_Pln.hxx"
class Plane : public Primitive
{
    double p[3];
    double d[3];
    double x[2];
    double y[2];
public:
    Plane (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double p1=0, double p2=0, double p3=0,
                  double d1=0, double d2=0, double d3=1,
                  double x1=0, double x2=100,
                  double y1=0, double y2=100)
    {
        p[0] = p1;
        p[1] = p2;
        p[2] = p3;
        d[0] = d1;
        d[1] = d2;
        d[2] = d3;
        x[0] = x1;
        x[1] = x2;
        y[0] = y1;
        y[1] = y2;
        T = PrimType::plane;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetPlaneModule(p,d,x,y);
    }
};
// =======================================================================
// a class for cube with parameters for width, length, height
// and poistion and direction
// =======================================================================
class Cube : public Primitive
{
    double x[3];
    double p[3];
    double d[3];
public:
    Cube (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double x1=1, double x2=1, double x3=1,
                  double p1=0, double p2=0, double p3=0,
                  double d1=0, double d2=0, double d3=1)
    {
        x[0] = x1;
        x[1] = x2;
        x[2] = x3;
        p[0] = p1;
        p[1] = p2;
        p[2] = p3;
        d[0] = d1;
        d[1] = d2;
        d[2] = d3;
        T = PrimType::cube;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetBoxModule(x,p,d);
    }
};
// =======================================================================
// a class for sphere with parameters for radius and poistion
// =======================================================================
class Sphere : public Primitive {
    double radius;
    double p[3];
    double d[3];
public:
    Sphere (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double r = 1,
                  double p1 = 0, double p2 = 0, double p3 = 0,
                  double d1 = 0, double d2 = 0, double d3 = 1)
    {
        radius = r;
        p[0] = p1;
        p[1] = p2;
        p[2] = p3;
        d[0] = d1;
        d[1] = d2;
        d[2] = d3;
        T = PrimType::sphere;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetSphereModule(radius, p, d);
    }
};
// =======================================================================
// a class for cyliner with parameters for radius, height and poistion
// =======================================================================
class Cylinder : public Primitive
{
    double radius;
    double height;
    double p[3];
    double d[3];
public:
    Cylinder (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double r = 1, double h = 1,
                  double p1 = 0, double p2 = 0, double p3 = 0,
                  double d1 = 0, double d2 = 0, double d3 = 1)
    {
        radius = r;
        height = h;
        p[0] = p1;
        p[1] = p2;
        p[2] = p3;
        d[0] = d1;
        d[1] = d2;
        d[2] = d3;
        T = PrimType::cylinder;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetCylinderModule(radius, height, p, d);
    }
};
// =======================================================================
// a class for cone with parameters for radius, height and poistion
// =======================================================================
class Cone : public Primitive
{
    double radius1;
    double radius2;
    double height;
    double pos[3];
    double dir[3];
public:
    Cone (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double r1 = 0.5, double r2 = 0.0, double h = 1,
                  double p1 = 0, double p2 = 0, double p3 = 0,
                  double d1 = 0, double d2 = 0, double d3 = 1)
    {
        radius1 = r1;
        radius2 = r2;
        height = h;
        pos[0] = p1;
        pos[1] = p2;
        pos[2] = p3;
        dir[0] = d1;
        dir[1] = d2;
        dir[2] = d3;
        T = PrimType::cone;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetConeModule(radius1, radius2, height, pos, dir);
    }
};
// =======================================================================
// a class for cone with parameters for inner radius, outer radius and poistion
// =======================================================================
class Torus : public Primitive
{
    double radius1;
    double radius2;
    double pos[3];
    double dir[3];
public:
    Torus (TopoDS_Shape* S) : Primitive(S) {}
    void SetData (double r1 = 1, double r2 = 0.25,
                  double p1 = 0, double p2 = 0, double p3 = 0,
                  double d1 = 0, double d2 = 0, double d3 = 1)
    {
        radius1 = r1;
        radius2 = r2;
        pos[0] = p1;
        pos[1] = p2;
        pos[2] = p3;
        dir[0] = d1;
        dir[1] = d2;
        dir[2] = d3;
        T = PrimType::torus;
    }
    void ShowProperties(CADDockWidget* cad_dock)
    {
        cad_dock->SetTorusModule(radius1, radius2, pos, dir);
    }
};
// =======================================================================
// a data structure for primitives like cube, cylinder, torus, cone, sphere
// =======================================================================
#include "BRepAlgoAPI_Fuse.hxx"
class Primitives
{
private:
    std::vector<Primitive*> P;
public:
    int Include (TopoDS_Shape S)
    {
        for (int i = 0; i < P.size(); i++)
        {
            if (S.IsEqual(*(P[i]->Value()))) return i;
        }
        return -1;
    }
    void Add (Primitive* p)
    {
        P.push_back(p);
    }
    void Delete (Primitive* p)
    {
        int n = Include(*(p->Value()));
        if (n != -1)
            P.erase (P.begin() + n);
    }
    void Delete (TopoDS_Shape p)
    {
        int n = Include(p);
        if (n != -1)
            P.erase (P.begin() + n);
    }
    int size ()
    {
        return P.size();
    }
    Primitive* operator [] (int i)
    {
        return P[i];
    }
    void Clear ()
    {
        P.clear();
    }
    TopoDS_Shape* Union ()
    {
        if (P.size() == 0) return NULL;
        TopoDS_Shape* S = P[0]->Value();
        for (int i = 1; i < P.size(); i++)
        {
            S = new TopoDS_Shape(BRepAlgoAPI_Fuse(*S,*(P[i]->Value())).Shape());
        }
        return S;
    }
};
// =======================================================================
// boundary edges
// =======================================================================
#include "TopoDS_Iterator.hxx"
enum BoundaryType { Dirichlet, Neumann};
class Boundary
{
private:
    BoundaryType type;
    int id;
    TopoDS_Shape* S;
    double value[3];
public:
    bool meas_sel = false;
    Boundary (const TopoDS_Shape& _S)
    {
        S =  new TopoDS_Shape(_S);
        type = Neumann;
        id = 0;
        for (int i = 0; i < 3; i++)
            value[0] = 0;
    }
    void SetValue (double* t)
    {
        for (int i = 0; i < 3; i++)
            value[i] = t[i];
    }
    double* GetValue ()
    {
        return value;
    }
    void SetType (BoundaryType t = Dirichlet)
    {
        type = t;
    }
    void SetId (int i = 0)
    {
        id = i;
    }
    TopoDS_Shape* Value ()
    {
        return S;
    }
    BoundaryType Type ()
    {
        return type;
    }
    int Id ()
    {
        return id;
    }
};
#include "TopExp_Explorer.hxx"
class Boundaries
{
private:
    std::vector<Boundary*> P;
public:
    Boundaries () {}
    Boundaries (Primitives* Pr)
    {
        for (int i = 0; i < Pr->size(); i++)
        {
            Add(*((*Pr)[i]->Value()));
        }
    }
    void Reset (Primitives* Pr)
    {
        P.clear();
        for (int i = 0; i < Pr->size(); i++)
        {
            Add(*((*Pr)[i]->Value()));
        }
    }
    int Include (TopoDS_Shape S)
    {
        for (int i = 0; i < P.size(); i++)
        {
            if (S.IsEqual(*(P[i]->Value())))
            {
                return i;
            }
        }
        return -1;
    }
    void Add (const TopoDS_Shape& p)
    {
        if (p.ShapeType() == TopAbs_FACE)
        {
            TopExp_Explorer Ex;
            Ex.Init(p,TopAbs_EDGE);
            for (; Ex.More(); Ex.Next()) {
                Boundary* S = new Boundary(Ex.Current());
                P.push_back(S);
            }
        }
        else if (p.ShapeType() == TopAbs_SOLID || p.ShapeType() == TopAbs_COMPOUND)
        {
            TopExp_Explorer Ex;
            Ex.Init(p,TopAbs_FACE);
            for (; Ex.More(); Ex.Next()) {
                Boundary* S = new Boundary(Ex.Current());
                P.push_back(S);
            }
        }
    }
    void Delete (const TopoDS_Shape& p)
    {
        int n = Include(p);
        if (n != -1)
        {
            P.erase (P.begin() + n);
        }
    }
    int Size ()
    {
        return P.size();
    }
    Boundary* operator [] (int i)
    {
        return P[i];
    }
    void Clear ()
    {
        P.clear();
    }
    void ExportToFile ()
    {
        ofstream out;
        out.open(std::string("./BndConditions.txt").c_str());
        for (int i = 0; i < P.size(); i++)
        {
            if (P[i]->Type() == Dirichlet) out << 0 << " ";
            else if (P[i]->Type() == Neumann) out << 1 << " ";
            for (int j = 0; j < 3; j++)
            {
                out << (P[i]->GetValue())[j] << " ";
            }
            out << endl;
        }
        out << "END" << endl;
        out.close();
    }
};
#endif // PRIMITIVE_H
