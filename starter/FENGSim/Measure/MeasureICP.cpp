#include "MeasureICP.h"
//#include "pcl/point_types.h"

MeasureICP::MeasureICP()
{

}

void MeasureICP::run() {
    //PointCloud<PointXYZ>::Ptr src, tgt;
   // pcl::PointCloud<pcl::PointXYZ> cloud_src;
   // pcl::PointCloud<pcl::PointXYZ> cloud_tgt;

    /*
    std::ifstream is("./data/meas/fengsim_meas_model.vtk");
    const int len = 256;
    char L[len];
    int n = 0;
    while (is.getline(L,len)) {
    n++;
    }
    is.close();

    //pcl::PointCloud<pcl::PointXYZ> cloud_src;
    cloud_src.width    = n;
    cloud_src.height   = 1;
    cloud_src.is_dense = false;
    cloud_src.points.resize (cloud_src.width * cloud_src.height);
    is.open("./data/meas/fengsim_meas_model.vtk");
    int i = 0;
    while (is.getline(L,len)) {
    double z[3];
    sscanf(L,"%lf %lf %lf", z, z+1, z+2);
    cloud_src.points[i].x = z[0];
    cloud_src.points[i].y = z[1];
    cloud_src.points[i].z = z[2];
    i++;
    }
    is.close();
    //pcl::io::savePCDFile("cloud_src.pcd", cloud_src);



    is.open("./data/meas/fengsim_meas_scene.vtk");
    n = 0;
    while (is.getline(L,len)) {
    n++;
    }
    is.close();
    std::cout << n << std::endl;

    //pcl::PointCloud<pcl::PointXYZ> cloud_tgt;
    cloud_tgt.width    = n;
    cloud_tgt.height   = 1;
    cloud_tgt.is_dense = false;
    cloud_tgt.points.resize (cloud_tgt.width * cloud_tgt.height);
    is.open("./data/meas/fengsim_meas_scene.vtk");
    i = 0;
    while (is.getline(L,len)) {
    double z[3];
    sscanf(L,"%lf %lf %lf", z, z+1, z+2);
    cloud_tgt.points[i].x = z[0];
    cloud_tgt.points[i].y = z[1];
    cloud_tgt.points[i].z = z[2];
    i++;
    }
    is.close();

    src.reset (new PointCloud<PointXYZ>(cloud_tgt));
    tgt.reset (new PointCloud<PointXYZ>(cloud_src));

    //  if (loadPCDFile (argv[p_file_indices[0]], *src) == -1 || loadPCDFile (argv[p_file_indices[1]], *tgt) == -1)
    //{
    //print_error ("Error reading the input files!\n");
    //return (-1);
    //}





















    // Compute the best transformtion
    Eigen::Matrix4f transform;
    computeTransformation (src, tgt, transform);


    return 0;


    std::cerr << transform << std::endl;
    // Transform the data and write it to disk
    PointCloud<PointXYZ> output_init;
    transformPointCloud (*src, output_init, transform);
    savePCDFileBinary ("source_transformed.pcd", output_init);


    std::ofstream out("./output_init.vtk");
    for (int i = 0; i < output_init.size(); i++) {
      out << output_init.points[i].x << " ";
      out << output_init.points[i].y << " ";
      out << output_init.points[i].z << std::endl;
    }
    */
}
