//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////
////     the codes from /software/pcl/test/registration/test_registration.cpp
////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/pyramid_feature_matching.h>
#include <pcl/features/ppf.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/filters/voxel_grid.h>
// We need Histogram<2> to function, so we'll explicitly add kdtree_flann.hpp here
#include <pcl/kdtree/impl/kdtree_flann.hpp>
//(pcl::Histogram<2>)

using namespace pcl;
using namespace pcl::io;

class icp_align {
 public:
    Eigen::Matrix4f transform;
    Eigen::Matrix4f inverse_transform;
    double align (PointCloud<PointXYZ>& cloud_source, PointCloud<PointXYZ>& cloud_target, PointCloud<PointXYZ>& cloud_icp)
    {
        PointCloud<PointXYZ>::Ptr cloud_source_ptr, cloud_target_ptr;
	cloud_source_ptr = cloud_source.makeShared ();
	cloud_target_ptr = cloud_target.makeShared ();
	
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaximumIterations(1000);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(1e-5);
	
	icp.setInputSource(cloud_source_ptr);
	icp.setInputTarget(cloud_target_ptr);
	
	icp.align(cloud_icp);
	for (int i = 0; i < 4; i++)
	    for (int j = 0; j < 4; j++)
	        transform(i,j) = icp.getFinalTransformation()(i,j);

	std::cout << icp.getFinalTransformation() << std::endl;    
	// Check that we have sucessfully converged
	std::cout << "if icp conv: "<< icp.hasConverged() << std::endl;
	// Test that the fitness score is below acceptable threshold
	return icp.getFitnessScore();
    }

    void align_back (PointCloud<PointXYZ>& cloud_source, PointCloud<PointXYZ>& cloud_target, PointCloud<PointXYZ>& cloud_icp)
    {
        PointCloud<PointXYZ>::Ptr cloud_source_ptr, cloud_target_ptr;
	cloud_source_ptr = cloud_source.makeShared ();
	cloud_target_ptr = cloud_target.makeShared ();
	
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaximumIterations(500);
	icp.setTransformationEpsilon(1e-5);
	icp.setEuclideanFitnessEpsilon(1e-5);
	
	icp.setInputSource(cloud_source_ptr);
	icp.setInputTarget(cloud_target_ptr);
	
	icp.align(cloud_icp);
	
	std::cout << icp.getFinalTransformation() << std::endl;
	Eigen::Matrix4f trans;
	for (int i = 0; i < 4; i++)
	    for (int j = 0; j < 4; j++)
	        trans(i,j) = icp.getFinalTransformation()(i,j);
	inverse_transform = trans.inverse();
	transformPointCloud (cloud_target, cloud_icp, inverse_transform);
	
	// Check that we have sucessfully converged
	std::cout << "if icp conv: "<< icp.hasConverged() << std::endl;
	// Test that the fitness score is below acceptable threshold
	std::cout << icp.getFitnessScore() << std::endl;
    }
};

