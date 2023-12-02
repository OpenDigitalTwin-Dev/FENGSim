#include <pcl/point_cloud.h>
#include <pcl/common/utils.h> // pcl::utils::ignore
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <cmath>

using namespace pcl;
using namespace pcl::io;

using KdTreePtr = search::KdTree<PointXYZ>::Ptr;


int main (int argc, char** argv)
{
    PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);  
    double num = 200.0;
    //std::cout << M_PI << std::endl;
    for (int i=0; i<num+1; i++) {
        for (int j=0; j<2*num+1; j++) {
	    cloud->push_back(pcl::PointXYZ(sin(M_PI/num*i)*cos(2.0*M_PI/2.0/num*j), sin(M_PI/num*i)*sin(2.0*M_PI/2.0/num*j), cos(M_PI/num*i)));
	    //std::cout << cos(M_PI/num*i)*cos(2*M_PI/num*j) << " " << cos(M_PI/num*i)*sin(2*M_PI/num*j) << " " << sin(M_PI/num*i) << std::endl;
	}
    }

    /*double error = 0;
    std::cout << "points num: " << cloud->size() << std::endl;
    for (int i = 0; i < cloud->size(); i++)
    {
        error += abs((*cloud)[i].x);
    }
    std::cout << error << std::endl;*/
	

    //NormalEstimationOMP<PointXYZ, Normal> ne (16);
    NormalEstimation<PointXYZ, Normal> ne; 
    PointCloud<Normal>::Ptr normals (new PointCloud<Normal> ());
    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
    ne.setInputCloud (cloud);
    ne.setRadiusSearch (0.5);  
    ne.setSearchMethod (tree);
    ne.compute (*normals);


    //FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
    FPFHEstimationOMP<PointXYZ, Normal, FPFHSignature33> fpfh_est(16);
    fpfh_est.setSearchMethod (tree);
    fpfh_est.setRadiusSearch (0.5);
    PointCloud<FPFHSignature33> features;
    fpfh_est.setInputCloud (cloud);
    fpfh_est.setInputNormals (normals);
    time_t t1;
    time(&t1);
    fpfh_est.compute (features);
    time_t t2;
    time(&t2);
    std::cout << difftime(t2, t1) << std::endl;

    



    // Display and retrieve the shape context descriptor vector for the 0th point.
    pcl::FPFHSignature33 descriptor = features[1];
    std::cout << descriptor << std::endl;
    std::cout << features.size() << std::endl;
    std::cout << getFieldsList<FPFHSignature33>(features) << std::endl;
    pcl::visualization::PCLPlotter plotter("fpfh");
    //plotter.addFeatureHistogram(features,33);
    plotter.addFeatureHistogram<FPFHSignature33>(features, "fpfh", 50, "one_fpfh");
    plotter.plot();
    
    
}
