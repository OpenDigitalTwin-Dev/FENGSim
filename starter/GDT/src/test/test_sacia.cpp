#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include "odt_pcl.h"
#include "odt_pcl_tools.h"

using namespace pcl;
using namespace pcl::io;

PointCloud<PointXYZ> cloud_source, cloud_target, cloud_reg;

void test_sacia () {
    // ##################################################
    int n = 10;
    double pi = 4.0*atan(1.0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; i<n/2+1; i++) {
        cloud_source->push_back(pcl::PointXYZ(0+i*1.0/n, 0, 0));
    }
    pcl::PointCloud<pcl::Normal>::Ptr cloud_source_normals (new pcl::PointCloud<pcl::Normal>);
    for (int i=0; i<n/2+1; i++) {
        cloud_source_normals->push_back(pcl::Normal(sin(i*pi/2/n), 0, cos(i*pi/2/n)));
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; i<n+1; i++) {
        cloud_target->push_back(pcl::PointXYZ(0+i*1.0/n, 0, 0));
    }
    pcl::PointCloud<pcl::Normal>::Ptr cloud_target_normals (new pcl::PointCloud<pcl::Normal>);
    for (int i=0; i<n+1; i++) {
        cloud_target_normals->push_back(pcl::Normal(sin(i*pi/2/n), 0, cos(i*pi/2/n)));
    }
    std::cout << "cloud source " << std::endl;
    std::cout << "------------------------------------" << std::endl;
    for (int i=0; i<cloud_source->size(); i++)
        std::cout << (*cloud_source)[i].x << " "
		  << (*cloud_source)[i].y << " "
		  << (*cloud_source)[i].z << " "
		  << std::endl;
    std::cout << std::endl;
    std::cout << "cloud source normal" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    for (int i=0; i<cloud_source_normals->size(); i++)
        std::cout << (*cloud_source_normals)[i].normal_x << " "
		  << (*cloud_source_normals)[i].normal_y << " "
		  << (*cloud_source_normals)[i].normal_z << " "
		  << std::endl;
    std::cout << std::endl;
    std::cout << "cloud target " << std::endl;
    std::cout << "------------------------------------" << std::endl;
    for (int i=0; i<cloud_target->size(); i++)
        std::cout << (*cloud_target)[i].x << " "
		  << (*cloud_target)[i].y << " "
		  << (*cloud_target)[i].z << " "
		  << std::endl;
    std::cout << std::endl;
    std::cout << "cloud target normal" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    for (int i=0; i<cloud_target_normals->size(); i++)
        std::cout << (*cloud_target_normals)[i].normal_x << " "
		  << (*cloud_target_normals)[i].normal_y << " "
		  << (*cloud_target_normals)[i].normal_z << " "
		  << std::endl;

    // ##################################################
    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
    FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
    fpfh_est.setSearchMethod (tree);
    fpfh_est.setRadiusSearch (0.5);
    PointCloud<FPFHSignature33> features_source, features_target;
    fpfh_est.setInputCloud (cloud_source);
    fpfh_est.setInputNormals (cloud_source_normals);
    fpfh_est.compute (features_source);
    fpfh_est.setInputCloud (cloud_target);
    fpfh_est.setInputNormals (cloud_target_normals);
    fpfh_est.compute (features_target);

    // ##################################################
    SampleConsensusInitialAlignment<PointXYZ, PointXYZ, FPFHSignature33> reg;
    std::cout << "getMinSampleDistance: " << reg.getMinSampleDistance() << std::endl;
    std::cout << "getMaxCorrespondenceDistance: " << reg.getMaxCorrespondenceDistance() << std::endl;
    /** \brief Get the minimum distances between samples, as set by the user */
    //reg.setMinSampleDistance (0.1f);
    /** \brief Get the maximum distance threshold between two correspondent points in source <-> target. If the 
     * distance is larger than this threshold, the points will be ignored in the alignment process.
     */
    //reg.setMaxCorrespondenceDistance (500);
    reg.setMaximumIterations(5); 
    reg.setCorrespondenceRandomness(3);
    //reg.setTransformationEpsilon (2);
    reg.setInputSource (cloud_source);
    reg.setInputTarget (cloud_target);
    reg.setSourceFeatures (features_source.makeShared ());
    reg.setTargetFeatures (features_target.makeShared ());
    // Register
    reg.align (cloud_reg);
    //export_pc_to_vtk(cloud_reg, "cloud_reg.vtk");
    std::cout << reg.getFitnessScore () << std::endl;

    std::cout << std::endl << "find similar feature" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << features_source.size() << " " << features_target.size() << std::endl;
    for (int i=0; i<features_source.size(); i++)
    {
	for (int j=0; j<features_target.size(); j++)
	{
	    pcl::FPFHSignature33 descriptor331 = features_source[i];
	    pcl::FPFHSignature33 descriptor332 = features_target[j];
	    double t = 0;
	    for (int k=0; k<33; k++)
	    {
	      t += pow(descriptor331.histogram[k]-descriptor332.histogram[k],2);
	    }
	    std::cout << t << " ";
	}
	std::cout << std::endl;
    }
}

