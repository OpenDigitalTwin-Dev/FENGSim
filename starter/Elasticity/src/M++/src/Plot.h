// file: Plot.h
// author: Christian Wieners, Martin Sauter
// $Header: /public/M++/src/Plot.h,v 1.22 2008-10-10 07:02:01 wieners Exp $

#ifndef _PLOT_H_
#define _PLOT_H_

#include "Mesh.h"
#include "IO.h"

class PlotData : public vector<double> {
    int i;
 public: 
    PlotData (int m=0, int id=-1) : vector<double>(m), i(id) {}
    int Id () const { return i; }
    void SetId (int j) { i = j; }
};

ostream& operator<< (ostream& s, const PlotData& p);

class plotdata : public hash_map<Point,PlotData,Hash>::const_iterator {
    typedef hash_map<Point,PlotData,Hash>::const_iterator Iterator;
 public:
    plotdata (Iterator p) : Iterator(p) {}
    const Point& operator () () const { return (*this)->first; }
    const PlotData& operator * () const { return (*this)->second; }
    double operator [] (int i) const { return (*this)->second[i]; }
    int Id () const { return (*this)->second.Id(); }
    int size () const { return (*this)->second.size(); }
};

ostream& operator<< (ostream& s, const plotdata& p);

class PlotVertexDatas : public hash_map<Point,PlotData,Hash> { 
 public:
    plotdata plotdatas () const { return plotdata(begin()); } 
    plotdata plotdatas_end () const { return plotdata(end()); } 
    plotdata find_plotdata (const Point& z) const { return plotdata(find(z)); }
    double operator () (const Point& x, int k) const;
    double max (int i = 0) const;
    
    double min (int i = 0) const;
    
    void Insert (const Point& z, int N);
    
    bool operator () (const Point& z) const;
    
    int Id (const Point& z) const;
    
    PlotData& data (const Point& z);
    
    const Point* numbering () const;
    
    const Point* numbering_data (int i) const;
    
    const Point* numbering_deformation_two (Point* z) const;
    
    const Point* numbering_deformation_three (Point* z) const;
    
    const Point* numbering_deformation (int d) const;
    
    void Renumbering ();
    
    template <class D> void data (const Mesh& M, const D& d, int shift=0);
    
    PlotVertexDatas (const Mesh& M, int N);
    
};

class PlotCell : public vector<int> {
 public:
    Point operator () (const Point* z) const;
};

class PlotCells : public vector<PlotCell> {
 public:
    PlotCells (const Mesh& M, const PlotVertexDatas& P);
};

class PlotCellDatas : public vector<PlotData> { 
 public:
    PlotCellDatas (int n, int N);
    template <class D> void data (const Mesh& M, const D& d, int shift=0);
    // for cell vectors - not working so far
    template <class D> void vtkdata (const Mesh& M, const D& d, int shift=0);
    template <class D> void singledata(const Mesh& M, const D& d, int shift=0);
//    void data(const Mesh& 
};

class Vector;

class Plot {
    int gnuplot;
    int dxplot;
    int tecplot;
    int vtkplot;
 public:
    const Mesh& M;
    PlotVertexDatas VD; 
    PlotCells C;
    PlotCellDatas CD; 
 public:
    Plot (const Mesh& m, int N=1, int K=0) : 
	M(m), VD(M,N), C(M,VD), CD(C.size(),K),
	gnuplot(0), dxplot(0), tecplot(0), vtkplot(0) {
    	ReadConfig(Settings,"gnuplot",gnuplot);    
	ReadConfig(Settings,"dxplot",dxplot);
	ReadConfig(Settings,"tecplot",tecplot);
	ReadConfig(Settings,"vtkplot",vtkplot);
    }
    int gnu_plot () const { return gnuplot; }
    int dx_plot () const { return dxplot; }
    int tec_plot () const { return tecplot; }
    int vtk_plot () const { return vtkplot; }
    const Mesh& GetMesh () const { return M; }
    const PlotVertexDatas& GetPlotVertexDatas () const { return VD; }
    const PlotCells& GetPlotCells () const { return C; }
    const PlotCellDatas& GetPlotCellDatas () const { return CD; }
    double max (int i=0) const { return VD.max(i); }
    double min (int i=0) const { return VD.min(i); }
    template <class D> void celldata (const D& d, int shift=0) {
	CD.data(M,d,shift); }
    template <class D> void singlecelldata (const D& d, int shift=0) {
	CD.singledata(M,d,shift); }
    template <class D> void vertexdata (const D& d, int shift=0) {
	VD.data(M,d,shift); }
    void vertexdata (const Vector&, int m = 1, int shift = 0);
    void celldata(const Vector&, int m = 1, int shift = 0);
    int nV () const { return VD.size(); }
    int nC () const { return CD.size(); }
    friend ostream& operator << (ostream& s, const Plot& plot);
 public:
    void gnu_mesh (const char* name);
    
    void gnu_vertexdata (const char* name, int i = 0);
    
    void gnu_deformation (const char* name);
    
    void gnu_vertexdata (const char* name, const Vector&, int k = 0);
 private:
    void vtk_mesh (ostream& out, int deform = 0);
    
    void dx_mesh (ostream& out, bool deformed = false);
    
    void vtk_vector(ostream& out, int k=0);

    void vtk_tensor(ostream& out);
    
    void vtk_cellvector(ostream& out, int k=0);    

    void vtk_celltensor(ostream& out, int k=0);

    void vtk_scalar(ostream& out, int k=0);
    
    void vtk_cell_data(ostream& out);
    
    void vtk_point_data(ostream& out);
    
    void vtk_2d_graph (ostream& out, int k =0);
 public:
    void vtk_vertex_vector (const char* name, int k = 0, int deformed = 0);

    void vtk_vertex_tensor(const char* name, int deformed);
    
    void vtk_vertexdata (const char* name, int k = 0, bool deformed = false);

    void vtk_vertexdata_smoothing (const char* name, int k = 0, bool deformed = false);
    
    void vtk_stokes (const char* name, int deform = 0);

    void vtk_cosserat (const char* name, int shift = 0, int deform = 0);    

    void vtk_special (const char* name, int deform, int k);

    void vtk_celldata (const char* name, int k = 0, bool deformed = false);

    void vtk_celldata (ostream& out, int k = 0, bool deformed = false);

    void vtk_celltensor(const char* name, int k=0, bool deformed = false);

    void vtk_cellvector(const char* name, int k=0, bool deformed = false);
    
    void vtk_2d_graph (const char* name, int k =0);

    void dx_mesh (const char* name, bool deformed = false);
    
    void dx_vertexdata_short (const char* name, int k = 0);
    
    void dx_vertexdata (const char* name, int k = 0, bool deformed = false);
    
    void dx_celldata (const char* name, int k = 0, bool deformed = false);
    
    template <class D> void dx_celldata(const char* name, const D& d, int k=0);
    
    void dx_load (const char* name = "load");
    void dx_vertexdata (const char* name, const Vector&, int k = 0);
    double operator () (const Point& x, int k) const { return VD(x,k); }
    bool operator () (const Point& x) const { return VD(x); } 
};

#endif
