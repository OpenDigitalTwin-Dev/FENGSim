#ifndef MESHGENERATION1_H
#define MESHGENERATION1_H

#include "Gmsh.h"
#include "GModel.h"
//#include "GEntity.h"
//#include "discreteFace.h"
//#include "discreteRegion.h"
//#include "MTriangle.h"
//#include "MFace.h"
#include <iostream>

#include "TopoDS_Shape.hxx"
#include <QString>

using namespace std;

#include "Slices.h"

class Voxel {
public:
    double x,y;
    Voxel (double _x, double _y) {
        x = _x;
        y = _y;
    }
};

class VoxelMesh : public vector<Voxel> {
public:
};

class Polygon {
    vector<double> points;
public:
    void AddPoint (double x, double y) {
        points.push_back(x);
        points.push_back(y);
    }
    double Area (double x1, double y1, double x2, double y2, double x3, double y3) {
        return abs (x1*y2*1 + x2*y3*1 + x3*y1*1 - x1*y3*1 - x2*y1*1 - x3*y2*1);
    }
    bool IsInside (Voxel v)
    {
        double x1 = points[0];
        double y1 = points[1];
        double x2 = points[2];
        double y2 = points[3];
        double x3 = points[4];
        double y3 = points[5];
        double x = v.x;
        double y = v.y;
        double area1 = Area(x1,y1,x2,y2,x,y);
        double area2 = Area(x3,y3,x2,y2,x,y);
        double area3 = Area(x1,y1,x3,y3,x,y);
        double area = Area(x1,y1,x2,y2,x3,y3);
        double _area = area1 + area2 + area3;
        if (abs(area-_area)<1e-10) {
            return true;
        }
        return false;
    }
};

class MeshModule {
    Slices slices;
    vector<Polygon> poly;
    vector<VoxelMesh> voxel_mesh;
    double voxel_h = 0.1;
    int mesh_dim;
public:
    MeshModule () {}
    void MeshGeneration (TopoDS_Shape* S, double size = 0.5, int refine_level = 0, QString path ="");
    void FileFormat ();
    void MeasureModel (QString path, QString filename = QString("/data/meas/fengsim_meas_target.vtk"));
    void MeasureModel (int id);
    bool IsInclude (std::vector<double> p, double x, double y, double z) {
        int n = p.size()/3;
        for (int i = 0; i < n; i++) {
            if (p[i*3]==x && p[i*3+1]==y && p[i*3+2]==z)
                return true;
        }
        return false;
    }
    void setdim (int n) {mesh_dim = n;}
    int getdim () {return mesh_dim;}
    void FileFormatCliToVTK (QString filename);
    void FileFormatMeshToVTK (QString Ain, QString Aout);
    void FileFormatMeshToVTK2 (QString Ain, QString Aout);
    void FileFormatMeshToGeo (QString Ain, QString Aout);
    void ClearSlices ()
    {
        slices.Clear();
    }
    int voxel_points () {
        int n = 0;
        for (int i = 0; i < voxel_mesh.size()-1; i++)
            n += voxel_mesh[i].size();
        return n;
    }
    void VoxelMeshGeneration ();
    void PathPlanning ();
};


#endif // MESHGENERATION1_H
