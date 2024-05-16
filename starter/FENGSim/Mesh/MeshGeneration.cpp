#include "MeshGeneration.h"
#include "GEdge.h"
#include "TopExp_Explorer.hxx"
#include "QString"

void MeshModule::MeshGeneration (TopoDS_Shape* S, double size, int refine_level, QString path)
{
    if (S == NULL) return;
    printf("USING GMSH\n");
    GmshInitialize();
    GmshSetOption("Mesh","Algorithm",2.);
    //GmshSetOption("Mesh","MinimumCurvePoints",20.);
    GmshSetOption("Mesh","CharacteristicLengthMax", size);
    GmshSetOption("General","Verbosity", 100.);

    // Create Gmsh model and set the factory
    GModel *GM = new GModel;
    GM->setFactory("OpenCASCADE");
    GM->importOCCShape(S);



    // dimension
    mesh_dim = GM->getDim();
    cout << "mesh dim: " << mesh_dim << endl;
    GM->mesh(3);
    // refine mesh
    for (int i = 0; i < refine_level; i++)
        GM->refineMesh(2);


    std::cout << (path+QString("/data/mesh/fengsim_mesh.vtk")).toStdString() << std::endl;
    GM->writeVTK((path+QString("/data/mesh/fengsim_mesh.vtk")).toStdString());
    GM->writeMESH((path+QString("/data/mesh/fengsim_mesh.mesh")).toStdString());





    delete GM;
    GmshFinalize();
    printf("Done meshing\n");
}

#include "fstream"
#include <QFileDialog>

void MeshModule::FileFormat ()
{
//    QString filename =  QFileDialog::getSaveFileName(0,"Save Mesh",
//                                                     QString("/home/jiping/M++/Elasticity/conf/geo"),
//                                                     "Mesh files (*.geo);;", 0 , QFileDialog::DontUseNativeDialog);
    ifstream is;
    ofstream out;
    is.open(std::string("./data/mesh/fengsim_mesh.mesh").c_str());
    // set dim
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int n4 = 0;
    const int len = 256;
    char L[len];
    // out .geo file
    //out.open(filename.toStdString().c_str());
    out.open("../Elasticity/build/solver/conf/geo/fengsim_mesh.geo");
    out << "POINTS" << endl;
    // vertices
    for (int i = 0; i < 5; i++) is.getline(L,len);
    sscanf(L,"%d", &n1);
    for (int i = 0; i < n1; i++)
    {
        double z[3];
        is.getline(L,len);
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        out << z[0] << " " << z[1] << " " << z[2] << endl;
    }
    // edges
    for (int i = 0; i < 2; i++) is.getline(L,len);
    sscanf(L,"%d", &n2);
    for (int i = 0; i < n2; i++)
    {
        is.getline(L,len);
    }
    out << "CELLS" << endl;
    if (mesh_dim == 2)
    {
        // faces
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n3);
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
            double z[3];
            sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
            out << 3 << " " << 0 << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << endl;
        }
    }
    else if (mesh_dim == 3)
    {
        // faces
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n3);
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
        }
        // cells
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n4);
        for (int i = 0; i < n4; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
            out << 4 << " " << 0 << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << " " << z[3] - 1 << endl;
        }
    }
    is.close();
    is.open(std::string("./data/mesh/fengsim_mesh.mesh").c_str());
    // vertices
    for (int i = 0; i < 5; i++) is.getline(L,len);
    for (int i = 0; i < n1; i++)
    {
        is.getline(L,len);
    }
    out << "FACES" << endl;
    if (mesh_dim == 2)
    {
        for (int i = 0; i < 2; i++) is.getline(L,len);
        for (int i = 0; i < n2; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf", z, z + 1, z + 2);
            out << 2 << " " << z[2] << " " << z[0] - 1 << " " << z[1] - 1 << endl;
        }
    }
    if (mesh_dim == 3)
    {
        for (int i = 0; i < 2; i++) is.getline(L,len);
        for (int i = 0; i < n2; i++)
        {
            is.getline(L,len);
        }
        for (int i = 0; i < 2; i++) is.getline(L,len);
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
            out << 3 << " " << z[3] << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << endl;
        }
    }
    is.close();
    out.close();

}

void MeshModule::MeasureModel (QString path, QString filename)
{
    std::vector<double> points;
    ifstream is;
    is.open((path+QString("/data/mesh/fengsim_mesh.mesh")).toStdString().c_str());
    ofstream out;
    out.open((path+filename).toStdString().c_str());

    int n = 0;
    const int len = 256;
    char L[len];
    for (int i = 0; i < 5; i++) is.getline(L,len);
    sscanf(L,"%d", &n);

    for (int i = 0; i < n; i++)
    {
        double z[3];
        is.getline(L,len);
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        out << setprecision(5) << z[0] << " " << z[1] << " " << z[2] << endl;
    }
    out.close();
    is.close();
}

void MeshModule::MeasureModel (int id)
{
    //    QString filename =  QFileDialog::getSaveFileName(0,"Save Measure Model",
    //                                                     QString("/home/jiping/FENGSim/FENGSim/Measure/data"),
    //                                                     "Measure files (*.vtk);;", 0 , QFileDialog::DontUseNativeDialog);
    std::vector<double> points;
    ifstream is;
    ofstream out;
    is.open(std::string("/home/jiping/FENGSim/build-FENGSim-Desktop_Qt_5_10_1_GCC_64bit-Release/FENGSimDT.mesh").c_str());
    // set dim
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int n4 = 0;
    const int len = 256;
    char L[len];
    // out .geo file
    out.open(std::string("/home/jiping/FENGSim/FENGSim/Measure/data/cloud_Model.vtk").c_str(),ios::app);
    // out << "POINTS" << endl;
    // vertices
    for (int i = 0; i < 5; i++) is.getline(L,len);
    sscanf(L,"%d", &n1);
    for (int i = 0; i < n1; i++)
    {
        double z[3];
        is.getline(L,len);
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
        //out << z[0] << " " << z[1] << " " << z[2] << endl;
        points.push_back(z[0]);
        points.push_back(z[1]);
        points.push_back(z[2]);
    }
    // edges
    for (int i = 0; i < 2; i++) is.getline(L,len);
    sscanf(L,"%d", &n2);
    for (int i = 0; i < n2; i++)
    {
        is.getline(L,len);
    }
    //out << "CELLS" << endl;
    if (mesh_dim == 2)
    {
        // faces
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n3);
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
            double z[3];
            sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
            //out << 3 << " " << 0 << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << endl;
        }
    }
    else if (mesh_dim == 3)
    {
        // faces
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n3);
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
        }
        // cells
        for (int i = 0; i < 2; i++) is.getline(L,len);
        sscanf(L,"%d", &n4);
        for (int i = 0; i < n4; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
            //out << 4 << " " << 0 << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << " " << z[3] - 1 << endl;
        }
    }
    is.close();
    is.open(std::string("/home/jiping/FENGSim/build-FENGSim-Desktop_Qt_5_10_1_GCC_64bit-Release/FENGSimDT.mesh").c_str());
    // vertices
    for (int i = 0; i < 5; i++) is.getline(L,len);
    for (int i = 0; i < n1; i++)
    {
        is.getline(L,len);
    }
    //out << "FACES" << endl;
    if (mesh_dim == 1)
    {
        for (int i = 0; i < 2; i++) is.getline(L,len);
        for (int i = 0; i < n2; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf", z, z + 1, z + 2);
            //out << 2 << " " << z[2] << " " << z[0] - 1 << " " << z[1] - 1 << endl;
        }
    }
    if (mesh_dim == 2)
    {
        for (int i = 0; i < 2; i++) is.getline(L,len);
        for (int i = 0; i < n2; i++)
        {
            is.getline(L,len);
        }
        for (int i = 0; i < 2; i++) is.getline(L,len);
        std::vector<double> pp;
        for (int i = 0; i < n3; i++)
        {
            is.getline(L,len);
            double z[4];
            sscanf(L,"%lf %lf %lf %lf", z, z + 1, z + 2, z + 3);
            //out << 3 << " " << z[3] << " " << z[0] - 1 << " " << z[1] - 1 << " " << z[2] - 1 << endl;
            if (z[3] == id) {
                //std::cout << z[0] << " " << z[1] << " " << z[2] << " " << z[3] << endl;
                if (!IsInclude(pp,points[(z[0]-1)*3],points[(z[0]-1)*3+1],points[(z[0]-1)*3+2])) {
                    pp.push_back(points[(z[0]-1)*3]);
                    pp.push_back(points[(z[0]-1)*3+1]);
                    pp.push_back(points[(z[0]-1)*3+2]);
                    out << points[(z[0]-1)*3] << " " << points[(z[0]-1)*3+1] << " " << points[(z[0]-1)*3+2] << endl;
                }
                if (!IsInclude(pp,points[(z[1]-1)*3],points[(z[1]-1)*3+1],points[(z[1]-1)*3+2])) {
                    pp.push_back(points[(z[1]-1)*3]);
                    pp.push_back(points[(z[1]-1)*3+1]);
                    pp.push_back(points[(z[1]-1)*3+2]);
                    out << points[(z[1]-1)*3] << " " << points[(z[1]-1)*3+1] << " " << points[(z[1]-1)*3+2] << endl;
                }
                if (!IsInclude(pp,points[(z[2]-1)*3],points[(z[2]-1)*3+1],points[(z[2]-1)*3+2])) {
                    pp.push_back(points[(z[2]-1)*3]);
                    pp.push_back(points[(z[2]-1)*3+1]);
                    pp.push_back(points[(z[2]-1)*3+2]);
                    out << points[(z[2]-1)*3] << " " << points[(z[2]-1)*3+1] << " " << points[(z[2]-1)*3+2] << endl;
                }
            }
        }
    }

}

#include "Slices.h"
void MeshModule::FileFormatCliToVTK (QString filename)
{
    ifstream is;
    is.open(filename.toStdString().c_str());

    const int len = 500000;
    char L[len];
    for (int i = 0; i < 7; i++)
        is.getline(L,len);
    double height;
    while(is.getline(L,len))
    {
        if (strncasecmp("$$LAYER",L,7) == 0) {
            sscanf(L, "$$LAYER/%lf", &height);
        }
        if (strncasecmp("$$POLYLINE",L,10) == 0) {
            slices.InsertSlice(height);
            int z[3];
            sscanf(L, "$$POLYLINE/%d,%d,%d", z, z + 1, z + 2);
            for (int i = 0; i < z[2]; i++) {
                double t[2];
                QString ss("$$POLYLINE/%*d,%*d,%*d");
                for (int j = 0; j < i*2; j++)
                {
                    ss += QString(",%*lf");
                }
                ss += QString(",%lf,%lf");
                sscanf(L, ss.toStdString().c_str(), t, t + 1);
                slices.InsertXY(t[0],t[1]);
            }
        }
    }
    is.close();
    ofstream out;
    out.open(std::string("/home/jiping/FENGSim/FENGSim/data/slices.vtk").c_str());
    int n = 0;
    for (int i = 0; i < slices.size(); i++)
        n += slices.SlicesNum(i);
    out <<"# vtk DataFile Version 2.0" << endl;
    out << "slices example" << endl;
    out << "ASCII" << endl;
    out << "DATASET POLYDATA" << endl;
    out << "POINTS " + QString::number(n).toStdString() + " float" << endl;
    for (int i = 0; i < slices.size(); i++)
    {
        for (int j = 0; j < slices.SlicesNum(i); j++)
        {
            out << slices.X(i,j) << " " << slices.Y(i,j) << " " << slices.SliceHeight(i) << endl;
        }
    }
    out << "POLYGONS " + QString::number(slices.size()).toStdString() + " " + QString::number(slices.size()+n).toStdString() << endl;
    int m = 0;
    for (int i = 0; i < slices.size(); i++) {
        out << slices.SlicesNum(i);
        for (int j = 0; j < slices.SlicesNum(i); j++) {
            out << " " << m + j;
        }
        out << endl;
        m += slices.SlicesNum(i);
    }
    slices.SetConvexHull();
}

void MeshModule::FileFormatMeshToVTK(QString Ain, QString Aout)
{
    ifstream is;
    ofstream out;
    is.open(Ain.toStdString().c_str());
    out.open(Aout.toStdString().c_str());

    out << "# vtk DataFile Version 2.0" << endl;
    out << "am mesh" << endl;
    out << "ASCII" << endl;
    out << "DATASET UNSTRUCTURED_GRID" << endl;
    const int len = 500;
    char L[len];
    for (int i = 0; i < 4; i++)
        is.getline(L,len);
    int vn;
    sscanf(L, "%d", &vn);
    out << "POINTS " + QString::number(vn).toStdString() + " float" << endl;
    for (int i = 0; i < vn; i++)
    {
        is.getline(L,len);
        double z[3];
        sscanf(L, "%lf %lf %lf", z, z + 1,z+2);
        out << z[0] << " " << z[1] << " " << z[2] << endl;
    }

    for (int i = 0; i < 2; i++)
        is.getline(L,len);
    int cn;
    sscanf(L, "%d", &cn);
    out << "CELLS " + QString::number(cn).toStdString() + " " + QString::number(cn*5).toStdString() << endl;

    for (int i = 0; i < cn; i++)
    {
        is.getline(L,len);
        int z[4];
        sscanf(L, "%d %d %d %d", z, z + 1, z + 2, z + 3);
        out << 4 << " " << z[0]-1 << " " << z[1]-1 << " " << z[2]-1 << " " << z[3]-1 << endl;
    }

    out << "CELL_TYPES " + QString::number(cn).toStdString() << endl;
    for (int i = 0; i < cn; i++)
    {
        out << 10 << endl;
    }
}

void MeshModule::FileFormatMeshToVTK2(QString Ain, QString Aout)
{
    ifstream is;
    ofstream out;
    is.open(Ain.toStdString().c_str());
    out.open(Aout.toStdString().c_str());

    out << "# vtk DataFile Version 2.0" << endl;
    out << "am mesh" << endl;
    out << "ASCII" << endl;
    out << "DATASET UNSTRUCTURED_GRID" << endl;
    const int len = 500;
    char L[len];
    for (int i = 0; i < 5; i++)
        is.getline(L,len);
    int vn;
    sscanf(L, "%d", &vn);
    out << "POINTS " + QString::number(vn).toStdString() + " float" << endl;
    for (int i = 0; i < vn; i++)
    {
        is.getline(L,len);
        double z[3];
        sscanf(L, "%lf %lf %lf", z, z + 1,z+2);
        out << z[0] << " " << z[1] << " " << z[2] << endl;
    }

    // edges
    for (int i = 0; i < 2; i++)
        is.getline(L,len);
    int n1;
    sscanf(L, "%d", &n1);
    for (int i = 0; i < n1; i++)
    {
        is.getline(L,len);
    }

    // faces
    for (int i = 0; i < 2; i++)
        is.getline(L,len);
    int n2;
    sscanf(L, "%d", &n2);
    for (int i = 0; i < n2; i++)
    {
        is.getline(L,len);
    }

    // cells
    for (int i = 0; i < 2; i++)
        is.getline(L,len);
    int cn;
    sscanf(L, "%d", &cn);
    out << "CELLS " + QString::number(cn).toStdString() + " " + QString::number(cn*5).toStdString() << endl;

    for (int i = 0; i < cn; i++)
    {
        is.getline(L,len);
        int z[4];
        sscanf(L, "%d %d %d %d", z, z + 1, z + 2, z + 3);
        out << 4 << " " << z[0]-1 << " " << z[1]-1 << " " << z[2]-1 << " " << z[3]-1 << endl;
    }

    out << "CELL_TYPES " + QString::number(cn).toStdString() << endl;
    for (int i = 0; i < cn; i++)
    {
        out << 10 << endl;
    }
}

void MeshModule::VoxelMeshGeneration()
{
    slices.SetConvexHull();
    double x_min = slices.x_min();
    double x_max = slices.x_max();
    double y_min = slices.y_min();
    double y_max = slices.y_max();
    int n1 = (x_max - x_min) / voxel_h;
    int n2 = (y_max - y_min) / voxel_h;
    for (int j = 0; j < slices.size(); j++)
    {
        // define polygon
        Polygon _poly;
        for (int i = 0; i < slices.SlicesNum(j)-1; i++)
        {
            _poly.AddPoint(slices.X(j,i),slices.Y(j,i));
        }
        poly.push_back(_poly);
    }
    for (int k = 0; k < slices.size(); k++)
    {
        // generate voxel mesh
        VoxelMesh _voxel_mesh;
        for (int i = 0; i < n1+1; i++) {
            for (int j = 0; j < n2+1; j++) {
                Voxel v(x_min+i*voxel_h, y_min+j*voxel_h);
                if (poly[k].IsInside(v)) {
                    _voxel_mesh.push_back(v);
                }
            }
        }
        voxel_mesh.push_back(_voxel_mesh);
    }
    // vtk export
    ofstream out;
    out.open(std::string("/home/jiping/FENGSim/FENGSim/data/am_voxel_mesh.vtk").c_str());
    out << "# vtk DataFile Version 2.0" << endl;
    out << "am voxel mesh" << endl;
    out << "ASCII" << endl;
    out << "DATASET UNSTRUCTURED_GRID" << endl;
    int n = voxel_points();
    out << "POINTS " + QString::number(n*15).toStdString() + " float" << endl;
    for (int j = 0; j < voxel_mesh.size()-1; j++) {
        for (int i = 0; i < voxel_mesh[j].size(); i++)
        {
            double x = voxel_mesh[j][i].x;
            double y = voxel_mesh[j][i].y;
            //            out << x - voxel_h/2 << " " << y - voxel_h/2 << " " << slices.SliceHeight(j) << endl;
            //            out << x + voxel_h/2 << " " << y - voxel_h/2 << " " << slices.SliceHeight(j) << endl;
            //            out << x + voxel_h/2 << " " << y + voxel_h/2 << " " << slices.SliceHeight(j) << endl;
            //            out << x - voxel_h/2 << " " << y + voxel_h/2 << " " << slices.SliceHeight(j) << endl;
            //            out << x << " " << y << " " << slices.SliceHeight(j) << endl;
            double a[15*3] = {0.00, 0.00, 0.00,
                              1.00, 0.00, 0.00,
                              1.00, 1.00, 0.00,
                              0.00, 1.00, 0.00,
                              0.00, 0.00, 1.00,
                              1.00, 0.00, 1.00,
                              1.00, 1.00, 1.00,
                              0.00, 1.00, 1.00,
                              0.50, 0.50, 0.00,
                              0.50, 0.00, 0.50,
                              1.00, 0.50, 0.50,
                              0.50, 1.00, 0.50,
                              0.00, 0.50, 0.50,
                              0.50, 0.50, 1.00,
                              0.50, 0.50, 0.50
                             };
            for (int i = 0; i < 15*3; i++)
            {
                a[i] *= voxel_h;
            }
            double dir[3];
            dir[0] = x - voxel_h / 2;
            dir[1] = y - voxel_h / 2;
            dir[2] = slices.SliceHeight(j) + voxel_h / 2 - voxel_h / 2;
            for (int i = 0; i < 15; i++)
            {
                a[i*3] += dir[0];
                a[i*3+1] += dir[1];
                a[i*3+2] += dir[2];
                out << a[i*3] << " " << a[i*3+1] << " " << a[i*3+2] << endl;
            }
        }
    }
    out << "CELLS " + QString::number(n*24).toStdString() + " " + QString::number(n*24*5).toStdString() << endl;
    int b[24*4] = { 14, 8, 0, 1,
                    14, 8, 1, 2,
                    14, 8, 2, 3,
                    14, 8, 3, 0,
                    14, 9, 0, 1,
                    14, 9, 1, 5,
                    14, 9, 5, 4,
                    14, 9, 4, 0,
                    14, 10, 1, 2,
                    14, 10, 2, 6,
                    14, 10, 6, 5,
                    14, 10, 5, 1,
                    14, 11, 2, 3,
                    14, 11, 3, 7,
                    14, 11, 7, 6,
                    14, 11, 6, 2,
                    14, 12, 3, 0,
                    14, 12, 0, 4,
                    14, 12, 4, 7,
                    14, 12, 7, 3,
                    14, 13, 4, 5,
                    14, 13, 5, 6,
                    14, 13, 6, 7,
                    14, 13, 7, 4 };
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 24; j++) {
            out << 4 << " " << b[0 + j*4] + i*15 << " " << b[1 + j*4] + i*15 << " " << b[2 + j*4] + i*15 << " " << b[3 + j*4] + i*15 << endl;
        }
    }
    out << "CELL_TYPES " + QString::number(n*24).toStdString() << endl;
    for (int i = 0; i < n*24; i++)
    {
        out << 10 << endl;
    }
}

void MeshModule::PathPlanning() {
    ofstream out;
    out.open(std::string("/home/jiping/FENGSim/FENGSim/data/am_path_planning.vtk").c_str());
    int n = 0;
    for (int i = 0; i < voxel_mesh.size()-1; i++)
    {
        n += voxel_mesh[i].size();
    }
    out << "# vtk DataFile Version 2.0" << endl;
    out << "slices example" << endl;
    out << "ASCII" << endl;
    out << "DATASET POLYDATA" << endl;
    out << "POINTS " + QString::number(n*2).toStdString() + " float" << endl;
    for (int i = 0; i < voxel_mesh.size()-1; i++)
    {
        for (int j = 0; j < voxel_mesh[i].size(); j++)
        {
            double x1 = voxel_mesh[i][j].x - voxel_h/2;
            double x2 = voxel_mesh[i][j].x + voxel_h/2;
            double y1 = voxel_mesh[i][j].y;
            double y2 = y1;
            out << x1 << " " << y1 << " " << slices.SliceHeight(i) + voxel_h/2 << endl;
            out << x2 << " " << y2 << " " << slices.SliceHeight(i) + voxel_h/2 << endl;
        }
    }
    out << "POLYGONS " << n << " " << 3*n << endl;
    int m = 0;
    for (int i = 0; i < voxel_mesh.size()-1; i++)
    {
        for (int j = 0; j < voxel_mesh[i].size(); j++)
        {
            out << 2 << " " << 2*m << " " << 2*m + 1 << endl;
            m++;
        }
    }
}

void MeshModule::FileFormatMeshToGeo(QString Ain, QString Aout)
{
    ifstream is;
    ofstream out;
    is.open(Ain.toStdString().c_str());
    out.open(Aout.toStdString().c_str());

    out << "POINTS:" << endl;
    const int len = 500;
    char L[len];
    for (int i = 0; i < 4; i++)
        is.getline(L,len);
    int vn;
    sscanf(L, "%d", &vn);
    for (int i = 0; i < vn; i++)
    {
        is.getline(L,len);
        double z[3];
        sscanf(L, "%lf %lf %lf", z, z + 1,z+2);
        out << z[0] << " " << z[1] << " " << z[2] << endl;
    }

    out << "CELLS: " << endl;
    for (int i = 0; i < 2; i++)
        is.getline(L,len);
    int cn;
    sscanf(L, "%d", &cn);
    for (int i = 0; i < cn; i++)
    {
        is.getline(L,len);
        int z[4];
        sscanf(L, "%d %d %d %d", z, z + 1, z + 2, z + 3);
        out << 4 << " " << 0 << " " << z[0]-1 << " " << z[1]-1 << " " << z[2]-1 << " " << z[3]-1 << endl;
    }
}
