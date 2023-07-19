#include "AMConfig.h"
#include "iostream"

AMConfig::AMConfig()
{

}

void AMConfig::clear (std::string filename)
{
    QDir dir("./../../AM/data/vtk");
    QStringList stringlist_vtk;
    //stringlist_vtk << "am_mesh_*.vtk";
    stringlist_vtk << filename.c_str();
    dir.setNameFilters(stringlist_vtk);
    QFileInfoList fileinfolist;
    fileinfolist = dir.entryInfoList();
    for (int i = 0; i < fileinfolist.size(); i++) {
        QString filename = fileinfolist.at(i).filePath();
        dir.remove(filename);
    }
}

void AMConfig::clear()
{
    clear("am_mesh_*.vtk");
    clear("am_current_pos_*.vtk");
}

void AMConfig::reset ()
{
    std::ofstream out;
    out.open("./../../AM/conf/m++conf");
    out << "#loadconf = Poisson/conf/poisson.conf;" << std::endl;
    out << "#loadconf = Elasticity/conf/m++conf;" << std::endl;
    out << "#loadconf = ElastoPlasticity/conf/m++conf;" << std::endl;
    out << "#loadconf = ThermoElasticity/conf/m++conf;" << std::endl;
    out << "loadconf = AdditiveManufacturing/conf/m++conf;" << std::endl;
    out << "loadconf = Cura/conf/m++conf;" << std::endl;
    out.close();


    out.open("./../../AM/AdditiveManufacturing/conf/am.conf");
    out << "Model = AM;" << std::endl;
    out << "GeoPath = AdditiveManufacturing/;" << std::endl;

    out << "SourceV = " << am_source_v <<";" << std::endl;
    out << "SourceX = " << am_source_x <<";" << std::endl;
    out << "SourceY = " << am_source_y <<";" << std::endl;
    out << "SourceZ = " << am_source_z <<";" << std::endl;
    out << "SourceH = " << am_source_h <<";" << std::endl;

    out << "Time = " << time <<";" << std::endl;
    out << "TimeSteps = " << time_num <<";" << std::endl;
    out << "TimeLevel = 2;" << std::endl;


    out << "Mesh = thinwall;" << std::endl;
    out << "Mesh2 = thinwall2;" << std::endl;
    out << "plevel = 0;" << std::endl;
    out << "level = 0;" << std::endl;



    out << "EXAMPLE = 1;" << std::endl;
    out << "Discretization = linear;" << std::endl;
    out << "Overlap_Distribution = 0;" << std::endl;
    out << "Overlap = none;" << std::endl;
    out << "Distribution = Stripes;" << std::endl;
    out << "NewtonSteps = 10;" << std::endl;
    out << "NewtonResidual = 1e-20;" << std::endl;

    out << "LinearSolver = CG;" << std::endl;
    out << "Preconditioner = Jacobi;" << std::endl;
    out << "LinearSteps = 50000;" << std::endl;
    out << "LinearEpsilon = 1e-15;" << std::endl;
    out << "LinearReduction = 1e-15;" << std::endl;

    out << "QuadratureCell = 3;" << std::endl;
    out << "QuadratureBoundary = 2;" << std::endl;

    out << "precision = 5;" << std::endl;
    out << "Young = 2.5;" << std::endl;
    out << "PoissonRatio = 0.25;" << std::endl;

    out.close();

}
