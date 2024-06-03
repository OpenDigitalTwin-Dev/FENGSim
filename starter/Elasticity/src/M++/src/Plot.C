// file: Plot.C
// author: Christian Wieners, Martin Sauter
// $Header: /public/M++/src/Plot.C,v 1.9 2008-10-10 07:02:01 wieners Exp $

#include "Plot.h"
#include "Algebra.h"
#include "Small.h"

ostream& operator<< (ostream& s, const PlotData& p) {
    for (int j=0; j<p.size(); ++j) s  << " " << p[j];
    return s << " 0 0 0 " << endl;
    return s << " id " << p.Id() << endl;
}

ostream& operator<< (ostream& s, const plotdata& p) {
    s << p();
    for (int j=0; j<p->second.size(); ++j) s  << " " << p[j];
    return s << " 0 0 0 " << endl;
    return s << " id " << p.Id() << endl;
}

double PlotVertexDatas::operator () (const Point& x, int k) const { 
    plotdata p = find_plotdata(x);
    return p[k]; 
}
double PlotVertexDatas::max (int i) const {
    double a = -infty;
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) 
        if (a < p[i]) a = p[i];
    return PPM->Max(a);
}

double PlotVertexDatas::min (int i) const {
    double a = infty;
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) 
        if (a > p[i]) a = p[i];
    return PPM->Min(a);
}
void PlotVertexDatas::Insert (const Point& z, int N) { 
    if (find(z) != end()) return;
    int i = size();
    (*this)[z] = PlotData(N,i); 
}

///////////////////////////////////////////////////////////////////////////

bool PlotVertexDatas::operator () (const Point& z) const { return (find(z)!=end()); }

int PlotVertexDatas::Id (const Point& z) const { return find_plotdata(z).Id(); }

PlotData& PlotVertexDatas::data (const Point& z) { 
    hash_map<Point,PlotData,Hash>::iterator p=find(z);
    return p->second;
}

const Point* PlotVertexDatas::numbering () const {
    Point* z = new Point [size()];
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) z[p.Id()] = p();
    return z;
}

const Point* PlotVertexDatas::numbering_data (int i) const {
    Point* z = new Point [size()];
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) 
        z[p.Id()] = Point(p()[0],p()[1],p[i]);
    return z;
}

const Point* PlotVertexDatas::numbering_deformation_two (Point* z) const {
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) 
        z[p.Id()] = Point(p()[0]+p[0],p()[1]+p[1]);
    return z;
}
const Point* PlotVertexDatas::numbering_deformation_three (Point* z) const {
    for (plotdata p=plotdatas(); p!=plotdatas_end(); ++p) 
        z[p.Id()] = Point(p()[0]+p[0],p()[1]+p[1],p()[2]+p[2]);
    return z;
}
const Point* PlotVertexDatas::numbering_deformation (int d) const {
    Point* z = new Point [size()];
    if (d == 2) return numbering_deformation_two(z);
    return numbering_deformation_three(z);
}
void PlotVertexDatas::Renumbering () {
    int i=0;
    for (hash_map<Point,PlotData,Hash>::iterator p=begin(); p!=end(); ++p) 
        p->second.SetId(i++);
}

template <class D> void PlotVertexDatas::data (const Mesh& M, const D& d, int shift) {
    ExchangeBuffer E;
    int m = d.size(); 
    for (vertex v=M.vertices(); v != M.vertices_end(); ++v) {
        E.Send(0) << v();
        for (int j=0; j<m; ++j) E.Send(0) << d(v(),j);
    }
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q)  
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            Point z;
            E.Receive(q) >> z;
            hash_map<Point,PlotData,Hash>::iterator p = find(z);
            for (int j=0; j<m; ++j) 
                E.Receive(q) >> p->second[j+shift];
        }
}

PlotVertexDatas::PlotVertexDatas (const Mesh& M, int N) {
    ExchangeBuffer E;
    for (vertex v=M.vertices(); v != M.vertices_end(); ++v)
        E.Send(0) << v();
    E.Communicate();
    for (short q=0; q<PPM->size(); ++q)  
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            Point x;
            E.Receive(q) >> x;
            Insert(x,N);
        }
    Renumbering();
}

///////////////////////////////////////////////////////////////////////////

Point PlotCell::operator () (const Point* z) const {
    Point x = zero;
    for (int i=0; i<size(); ++i) x += z[(*this)[i]];
    return (1.0 / size()) * x;
}

///////////////////////////////////////////////////////////////////////////

PlotCells::PlotCells (const Mesh& M, const PlotVertexDatas& P) {
    int n = PPM->Sum(int(M.Cells::size()));
    if (PPM->master()) resize(n);
    ExchangeBuffer E;
    for (cell c=M.cells(); c!=M.cells_end(); ++c) {
        E.Send(0) << short(c.Corners());
        for (int i=0; i<c.Corners(); ++i) E.Send(0) << c[i];
    }
    E.Communicate();
    n = 0;
    for (short q=0; q<PPM->size(); ++q)  
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            short m;
            E.Receive(q) >> m;
            (*this)[n].resize(m);
            Point z;
            for (int i=0; i<m; ++i) {
                E.Receive(q) >> z;
                (*this)[n][i] = P.Id(z);
            }
            ++n;
        }
}

///////////////////////////////////////////////////////////////////////////

PlotCellDatas::PlotCellDatas (int n, int N) : vector<PlotData>(n) {
    for (int i=0; i<n; ++i) (*this)[i] = PlotData(N);
}

template <class D> 
void PlotCellDatas::data (const Mesh& M, const D& d, int shift) {
    ExchangeBuffer E;
    int m = d.size();
    
    for (cell c=M.cells(); c!=M.cells_end(); ++c) {
	for (int j=0; j<m; ++j) E.Send(0) << d(c(),j);
    }
    E.Communicate();
    int i = 0;
    for (short q=0; q<PPM->size(); ++q)  
        while (E.Receive(q).size() < E.ReceiveSize(q)) {
            for (int j=0; j<m; ++j) {
                double a;
                E.Receive(q) >> a;
                (*this)[i][j+shift] = a;
            }
            ++i;
        }
}

template <class D> 
void PlotCellDatas::singledata (const Mesh& M, const D& d, int shift) {
	ExchangeBuffer E;
	for (cell c=M.cells(); c!=M.cells_end(); ++c) 
	    E.Send(0) << d(c());
	E.Communicate();
	int i = 0;
	for (short q=0; q<PPM->size(); ++q)  
	    while (E.Receive(q).size() < E.ReceiveSize(q)) {
		double a;
		E.Receive(q) >> a;
		(*this)[i][shift] = a;
		++i;
	    }
    }


ostream& operator << (ostream& s, const Plot& plot) {
    for (plotdata p=plot.VD.plotdatas(); p!=plot.VD.plotdatas_end(); ++p) 
        s << p;
    s << endl;
    for (int i=0; i<plot.C.size(); ++i, s << endl)
        for (int j=0; j<plot.C[i].size(); ++j) 
            s << plot.C[i][j] + 1 << " ";
    return s;
}

void Plot::gnu_mesh (const char* name) {
    if (!PPM->master()) return;
    string filename = string("data/gp/") + name + string(".gp");
    M_ofstream out(filename.c_str());  
    const Point* z = VD.numbering();
    out << z[C[0][0]] << endl;
    for (int i=0; i<C.size(); ++i) {
        if (C[i].size() == 3) {
            for (int j=0; j<C[i].size(); ++j)
                out << z[C[i][j]] << endl;
        }
        else if (C[i].size() == 4) {
            for (int j=0; j<C[i].size(); ++j)
                out << z[C[i][j]] << endl;
        }
        else if (C[i].size() == 8) {
            out << z[C[i][0]] << endl
                << z[C[i][1]] << endl
                << z[C[i][2]] << endl
                << z[C[i][3]] << endl
                << z[C[i][0]] << endl << endl
                << z[C[i][4]] << endl
                << z[C[i][5]] << endl
                << z[C[i][6]] << endl
                << z[C[i][7]] << endl
                << z[C[i][4]] << endl << endl
                << z[C[i][0]] << endl
                << z[C[i][4]] << endl << endl
                << z[C[i][1]] << endl
                << z[C[i][5]] << endl << endl
                << z[C[i][2]] << endl
                << z[C[i][6]] << endl << endl
                << z[C[i][3]] << endl
                << z[C[i][7]] << endl << endl;
        }
        out << z[C[i][0]] << endl << endl;
    }
    delete[] z;
}

void Plot::gnu_vertexdata (const char* name, int i) {
    if (!PPM->master()) return;
    string filename = string("data/gp/") + name + string(".gp");
    M_ofstream out(filename.c_str());  
    const Point* z = VD.numbering_data(i);
    Point_3d();
    out << z[C[0][0]] << endl;
    for (int i=0; i<C.size(); ++i) {
        for (int j=0; j<C[i].size(); ++j) 
            out << z[C[i][j]] << endl;
        out << z[C[i][0]] << endl << endl;
    }
    delete[] z;
    if (M.dim() == 2) Point_2d();
}

void Plot::gnu_deformation (const char* name) {
    if (!PPM->master()) return;
    string filename = string("data/gp/") + name + string(".gp");
    M_ofstream out(filename.c_str());  
    const Point* z = VD.numbering_deformation(M.dim());
    out << z[C[0][0]] << endl;
    for (int i=0; i<C.size(); ++i) {
        if (C[i].size() == 3) {
            for (int j=0; j<C[i].size(); ++j)
                out << z[C[i][j]] << endl;
        }
        else if (C[i].size() == 4) {
            for (int j=0; j<C[i].size(); ++j)
                out << z[C[i][j]] << endl;
        }
        else if (C[i].size() == 8) {
            out << z[C[i][0]] << endl
                << z[C[i][1]] << endl
                << z[C[i][2]] << endl
                << z[C[i][3]] << endl
                << z[C[i][0]] << endl << endl
                << z[C[i][4]] << endl
                << z[C[i][5]] << endl
                << z[C[i][6]] << endl
                << z[C[i][7]] << endl
                << z[C[i][4]] << endl << endl
                << z[C[i][0]] << endl
                << z[C[i][4]] << endl << endl
                << z[C[i][1]] << endl
                << z[C[i][5]] << endl << endl
                << z[C[i][2]] << endl
                << z[C[i][6]] << endl << endl
                << z[C[i][3]] << endl
                << z[C[i][7]] << endl << endl;
        }
        out << z[C[i][0]] << endl << endl;
    }
    delete[] z;
}

void Plot::vtk_mesh(ostream& out, int deform) {
    out<< "# vtk DataFile Version 2.0"<<endl
       << "Unstructured Grid by M++"<<endl
       << "ASCII"<<endl
       << "DATASET UNSTRUCTURED_GRID"<<endl
       << "POINTS "<<VD.size()<<" float"<<endl;
    if (M.dim() == 2) Point_2d(); 
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
        if (deform) {
            Point pp = p();
            for (int i=0;i<M.dim(); ++i) {
	        pp[i] += p[i];
	    }
            out << pp;
        }
        else out << p();
        if (M.dim() ==2) out <<" 0.00";
        out<<endl;
    }
    out<<"CELLS "<<C.size()<<" "<<(C[0].size()+1)*C.size()<<endl;
    for (int i=0; i<C.size(); ++i, out << endl) {
        out << C[i].size() <<" ";
        switch (C[i].size()) { 
        case 3: 
            out << C[i][0] << " " << C[i][1] << " " << C[i][2];
            break; 
        case 4: 
            if (M.dim() == 2) 
                out<<C[i][0]<<" "<<C[i][1]<<" "<<C[i][3]<<" "<< C[i][2];
            else
                out<<C[i][0]<<" "<<C[i][2]<<" "<<C[i][1]<<" "<< C[i][3];
            break; 
        case 8: 
            out << C[i][0] << " " << C[i][1] << " " << C[i][3]
                << " " << C[i][2] << " " << C[i][4]
                << " " << C[i][5] << " " << C[i][7] << " " << C[i][6];
            break; 
        }
    }
    out <<"CELL_TYPES "<<C.size()<<endl;
    for (int i=0; i<C.size(); ++i, out << endl) {
        switch (C[i].size()) { 
        case 3: out << "5";	
            break; 
        case 4: 
            if (M.dim() == 2) out<< "8";
            else out<< "10";
            break; 
        case 8: out << "11";
            break; 
        }
    }




	





	
}

void Plot::vtk_2d_graph (ostream& out, int k) {
    out<< "# vtk DataFile Version 2.0"<<endl
       << "Unstructured Grid by M++"<<endl
       << "ASCII"<<endl
       << "DATASET UNSTRUCTURED_GRID"<<endl
       << "POINTS "<<VD.size()<<" float"<<endl;
    Point_3d();
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
	Point x = p();
	x[2] = p[k];
	out << x;
    }
    out<<"CELLS "<<C.size()<<" "<<(C[0].size()+1)*C.size()<<endl;
    for (int i=0; i<C.size(); ++i, out << endl) {
        out << C[i].size() <<" ";
        switch (C[i].size()) { 
        case 3: 
            out << C[i][0] << " " << C[i][1] << " " << C[i][2];
            break; 
        case 4: 
	    out<<C[i][0]<<" "<<C[i][1]<<" "<<C[i][3]<<" "<< C[i][2];
            break; 
	}
    }
    out <<"CELL_TYPES "<<C.size()<<endl;
    for (int i=0; i<C.size(); ++i, out << endl) {
        switch (C[i].size()) { 
        case 3: out << "5";	
            break; 
        case 4: 
            if (M.dim() == 2) out<< "8";
            else out<< "10";
            break; 
        case 8: out << "11";
            break; 
        }
    }
}

void Plot::dx_mesh (ostream& out, bool deformed) {
    out << "object 1 class array type float rank 1 shape ";
    if (M.dim() == 2) out << "2"; 
    else out << "3"; 
    out << " items " << VD.size() << " data follows" << endl;
    if (deformed) {
        if (M.dim() == 2) {
            for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
                Point Def = Point(p()[0]+p[0],p()[1]+p[1]);
                out << Def << endl;
            }
        }
        else {
            for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
                Point Def = Point(p()[0]+p[0],p()[1]+p[1],p()[2]+p[2]);
                out << Def << endl;
            }
        }
    }	
    else {
        if (M.dim() == 2) Point_2d(); 
        for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p)
            out << p() << endl;
    }
    out << "object 2 class array type int rank 1 shape "
        << C[0].size() << " items "
        << C.size() << " data follows" << endl;
    for (int i=0; i<C.size(); ++i, out << endl) 
        switch (C[i].size()) { 
        case 3: 
            out << C[i][0] << " " << C[i][1] << " " << C[i][2];
            break; 
        case 4: 
            if (M.dim() == 2) 
                out<<C[i][0]<<" "<<C[i][1]<<" "<<C[i][3]<<" "<< C[i][2];
            else
                out<<C[i][0]<<" "<<C[i][2]<<" "<<C[i][1]<<" "<< C[i][3];
            break; 
        case 8: 
            out << C[i][0] << " " << C[i][1] << " " << C[i][3]
                << " " << C[i][2] << " " << C[i][4]
                << " " << C[i][5] << " " << C[i][7] << " " << C[i][6];
            break; 
        }
    out << "attribute \"element type\" string ";
    if (C[0].size() == 3) out << "\"triangles\"" << endl;
    else if (C[0].size() == 4) {
        if (M.dim() == 2) out << "\"quads\"" << endl;
        else       out << "\"tetrahedra\"" << endl;
    }
    else if (C[0].size() == 8) out << "\"cubes\"" << endl;
    out << "attribute \"ref\" string \"positions\"" << endl;
}

void Plot::vtk_vector(ostream& out, int k) {
    out << "VECTORS vector_value float"<<endl;


    double t = 0;

    
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {


        t += p[0] * p[0] + p[1] * p[1] + p[2] * p[2];


        if (abs(p[k]) < PlotTolerance) out << "0";
        else                           out << p[k];
        if (abs(p[k+1]) < PlotTolerance) out << " 0";
        else                           out << " "<<p[k+1];
        if (M.dim() == 2) out<<" 0"<<endl;
        else {
            if (abs(p[k+2]) < PlotTolerance) out << " 0"<<endl;
            else                           out << " "<<p[k+2]<<endl;
	}
    }

}

void Plot::vtk_tensor(ostream& out) {
    //out << "TENSORS tensor_value float"<<endl;
    for (int i=0; i<9; i++) {
	out << "SCALARS PRESSURE" << i << " float 1"<<endl;
	out << "LOOKUP_TABLE default"<<endl;

	double t = 0;

    
	for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {


	    t += p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
	    
	    
	    /*if (abs(p[k]) < PlotTolerance) out << "0";
	      else                           out << p[k];
	      if (abs(p[k+1]) < PlotTolerance) out << " 0";
	      else                           out << " "<<p[k+1];
	      if (M.dim() == 2) out<<" 0"<<endl;
	      else {
	      if (abs(p[k+2]) < PlotTolerance) out << " 0"<<endl;
	      else                           out << " "<<p[k+2]<<endl;
	      }*/

	    // 0 3 4
	    // 3 1 5
	    // 4 5 2
	    if (i==0) out << p[3] << endl;
	    if (i==1) out << p[6] << endl;
	    if (i==2) out << p[7] << endl;
	    if (i==3) out << p[6] << endl;
	    if (i==4) out << p[4] << endl;
	    if (i==5) out << p[8] << endl;
	    if (i==6) out << p[7] << endl;
	    if (i==7) out << p[8] << endl;
	    if (i==8) out << p[5] << endl;

	    //out << p[3] << " " << p[6] << " " << p[7] << endl;
	    //out << p[6] << " " << p[4] << " " << p[8] << endl;
	    //out << p[7] << " " << p[8] << " " << p[5] << endl << endl;
	}
    }

}

void Plot::vtk_cellvector(ostream& out, int k) {
    out << "VECTORS vector_value float"<<endl;
    for (int i=0; i<CD.size(); ++i) {
        if (abs(CD[i][k]) < PlotTolerance) out << "0";
        else                               out << CD[i][k];
        if (abs(CD[i][k+1]) < PlotTolerance) out << " 0";
        else                               out << " " << CD[i][k+1];
        if (abs(CD[i][k+2]) < PlotTolerance) out << " 0" << endl;
        else                               out << " " << CD[i][k+2] << endl;
    }
}

void Plot::vtk_celltensor(ostream& out, int k) {
    out << "TENSORS tensor_value float"<<endl;
    for (int i=0; i<CD.size(); ++i) {
        if (abs(CD[i][k]) < PlotTolerance) out << "0";
        else                               out << CD[i][k];
        if (abs(CD[i][k+3]) < PlotTolerance) out << " 0";
        else                               out << " "<<CD[i][k+3];
        if (abs(CD[i][k+4]) < PlotTolerance) out << " 0"<<endl;
        else                               out << " "<<CD[i][k+4]<<endl;

        if (abs(CD[i][k+3]) < PlotTolerance) out << "0";
        else                               out  << CD[i][k+3];
        if (abs(CD[i][k+1]) < PlotTolerance) out << " 0";
        else                               out << " " << CD[i][k+1];
        if (abs(CD[i][k+5]) < PlotTolerance) out << " 0"<<endl;
        else                               out << " " << CD[i][k+5]<<endl;

        if (abs(CD[i][k+4]) < PlotTolerance) out << "0";
        else                               out << CD[i][k+4];
        if (abs(CD[i][k+5]) < PlotTolerance) out << " 0";
        else                               out << " " << CD[i][k+5];
        if (abs(CD[i][k+2]) < PlotTolerance) out << " 0" << endl<<endl;
        else                               out << " " << CD[i][k+2] << endl<<endl;
    }
}


void Plot::vtk_scalar(ostream& out, int k) {
    out << "SCALARS scalar_value float 1"<<endl
        << "LOOKUP_TABLE default"<<endl;
    if (k==100) {
      for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
	  out << sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) << endl;
      }
      return;
    }
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
        if (abs(p[k]) < PlotTolerance) out << "0"<<endl;
	else                           out << p[k]<<endl;
    }
}

void Plot::vtk_cell_data(ostream& out) {
    out << "CELL_DATA " << CD.size() << endl;
}

void Plot::vtk_point_data (ostream& out) {
    out << "POINT_DATA " << VD.size() <<endl;
}


void Plot::vtk_vertex_vector(const char* name, int k, int deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deformed);
    vtk_point_data(out);
    vtk_vector(out,k);
}

void Plot::vtk_vertex_tensor(const char* name, int deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deformed);
    vtk_point_data(out);
    vtk_tensor(out);
}

void Plot::vtk_vertexdata(const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out, deformed);
    vtk_point_data(out);
    vtk_scalar(out,k);
}

void Plot::vtk_vertexdata_smoothing(const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out, deformed);

	out << "POINT_DATA "<< VD.size() <<endl;
	out << "SCALARS fixed float"<<endl
        << "LOOKUP_TABLE default"<<endl;
	for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
		Point pp = p();
		for (int i=0;i<M.dim(); ++i) {
			pp[i] += p[i];
		}
		if (pp[0]==0) {
			out << 1 << endl;
			continue;
		}
		else if (pp[0]==1) {
			out << 1 << endl;
			continue;
		}
		else if (pp[1]==0) {
			out << 1 << endl;
			continue;
		}
		else if (pp[1]==1) {
			out << 1 << endl;
			continue;
		}
		else
			out << 0 << endl;
    }
}

void Plot::vtk_2d_graph (const char* name, int k) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_2d_graph(out,k);
    vtk_point_data(out);
    vtk_scalar(out,k);
}

void Plot::vtk_celldata (ostream& out, int k, bool deformed) {
    vtk_mesh(out, deformed);
    vtk_cell_data(out);
    out << "SCALARS scalar_value float 1"<<endl
        << "LOOKUP_TABLE default"<<endl;
    for (int i=0; i<CD.size(); ++i) {
        if (abs(CD[i][k]) < PlotTolerance) out << "0" << endl;
        else                               out << CD[i][k] << endl;
    }
}

void Plot::vtk_celldata (const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_celldata(out,k,deformed);
}

void Plot::vtk_stokes(const char* name, int deform) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deform);
    vtk_point_data(out);
    vtk_scalar(out,M.dim());
    vtk_vector(out);
}

void Plot::vtk_cosserat(const char* name, int shift, int deform) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deform);
    vtk_cell_data(out);
    vtk_cellvector(out,shift);
}

void Plot::vtk_celltensor(const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deformed);
    vtk_cell_data(out);
    vtk_celltensor(out,k);
}

void Plot::vtk_cellvector(const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deformed);
    vtk_cell_data(out);
    vtk_cellvector(out,k);
}

void Plot::vtk_special(const char* name, int deform, int k) {
    if (!PPM->master()) return;
    string filename = string("data/vtk/") + name + string(".vtk");  
    M_ofstream out(filename.c_str());  
    vtk_mesh(out,deform);
    vtk_point_data(out);
    vtk_scalar(out,k);
    vtk_vector(out);
}

void Plot::dx_mesh (const char* name, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/dx/") + name + string(".dx");  
    M_ofstream out(filename.c_str());  
    dx_mesh(out,deformed);
}

void Plot::dx_vertexdata_short (const char* name, int k) {
    if (!PPM->master()) return;
    string filename = string("data/dx/") + name + string(".dx");  
    M_ofstream out(filename.c_str());  
    out << "object 3 class array type float rank 0 items "
        << VD.size() << " data follows" << endl;
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
        if (abs(p[k]) < PlotTolerance) out << "0" << endl;
        else                           out << p[k] << endl;
    }
    out << "attribute \"dep\" string \"positions\"" << endl 
        << "object \"simplex-part\" class field" << endl  
        << "component \"positions\" value 1" << endl  
        << "component \"connections\" value 2" << endl  
        << "component \"data\" value 3" << endl  
        << "end" << endl;
}
void Plot::dx_vertexdata (const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/dx/") + name + string(".dx");  
    M_ofstream out(filename.c_str());  
    dx_mesh(out,deformed);
    out << "object 3 class array type float rank 0 items "
        << VD.size() << " data follows" << endl;
    for (plotdata p=plotdata(VD.begin()); p!=plotdata(VD.end()); ++p) {
        if (abs(p[k]) < PlotTolerance) out << "0" << endl;
        else                           out << p[k] << endl;
    }
    out << "attribute \"dep\" string \"positions\"" << endl 
        << "object \"simplex-part\" class field" << endl  
        << "component \"positions\" value 1" << endl  
        << "component \"connections\" value 2" << endl  
        << "component \"data\" value 3" << endl  
        << "end" << endl;
}

void Plot::dx_celldata (const char* name, int k, bool deformed) {
    if (!PPM->master()) return;
    string filename = string("data/dx/") + name + string(".dx");  
    M_ofstream out(filename.c_str());  
    dx_mesh(out,deformed);
    out << "object 3 class array type float rank 0 items "
        << CD.size() << " data follows" << endl;
    for (int i=0; i<CD.size(); ++i) {
        if (abs(CD[i][k]) < PlotTolerance) out << "0" << endl;
        else                               out << CD[i][k] << endl;
    }
    out << "attribute \"dep\" string \"connections\"" << endl 
        << "object \"simplex-part\" class field" << endl  
        << "component \"positions\" value 1" << endl  
        << "component \"connections\" value 2" << endl  
        << "component \"data\" value 3" << endl  
        << "end" << endl;
}

template <class D> void Plot::dx_celldata(const char* name, const D& d, int k){
    celldata(d,k);
    dx_celldata(name);
}

///////////////////////////////////////////////////////////////////////////

double ProcId;
class ProcCellData {
 public:
    int size () const { return 1; }
    const double* operator () (const Point&, int) const { 
	ProcId = double(PPM->proc());
	return &ProcId;
    }
};
void Plot::dx_load (const char* name) { dx_celldata(name,ProcCellData()); }

class VectorVertexData {
    const Vector& u;
    int m;
 public:
    VectorVertexData (const Vector& U, int M = -1) : u(U), m(M) {
	if (m == -1) m = u.TypeDoF(0); }
    int size () const { return m; }
    double operator () (const Point& z, int j) const { 
	return double_of_Scalar(u(z,j)); 
    }
};

class VectorCellData {
    const Vector& u;
    int m;
public:
    VectorCellData(const Vector& U, int M = -1) : u(U), m(M) {}
    int size() const { return m; }
    double operator() (const Point& z, int j) const {
/*	mout <<" in VCData : z = "<<z<<"   ; j="<<j<<endl<<"u.size "<<u.size()
	     <<endl<<"u "<<u<<endl
	     <<"mgraph"<<endl<<u.GetMesh()<<endl;*/
//	Scalar a = u(z,j);
//	mout<<"aa "<<a<<endl;
	return double_of_Scalar(u(z,j));
    }
};

void Plot::vertexdata (const Vector& u, int m, int shift) {
    vertexdata(VectorVertexData(u,m),shift); }

void Plot::celldata  (const Vector& u, int m, int shift) {
    celldata(VectorCellData(u,m),shift);
}

void Plot::dx_vertexdata (const char* name, const Vector& u, int k) {
    vertexdata(VectorVertexData(u)); 
    dx_vertexdata(name,k); 
}
void Plot::gnu_vertexdata (const char* name, const Vector& u, int k) {
    vertexdata(VectorVertexData(u)); 
    gnu_vertexdata(name,k); 
}
