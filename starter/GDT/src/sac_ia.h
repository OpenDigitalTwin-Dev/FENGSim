//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////
////     the codes from /software/pcl/test/registration/test_sac_ia.cpp
////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d_omp.h>
#include "odt_pcl_tools.h"


using namespace pcl;
using namespace pcl::io;

class sacia_align {
 public:
    Eigen::Matrix4f transform;
    double align (PointCloud<PointXYZ>& cloud_source, PointCloud<PointXYZ>& cloud_target, PointCloud<PointXYZ>& cloud_sacia, double radius=1, double itnum=2000)
    {
        // Create shared pointers
        PointCloud<PointXYZ>::Ptr cloud_source_ptr, cloud_target_ptr;
	cloud_source_ptr = cloud_source.makeShared ();
	cloud_target_ptr = cloud_target.makeShared ();
	
	// Initialize estimators for surface normals and FPFH features
	search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
	
	NormalEstimation<PointXYZ, Normal> norm_est;
	//NormalEstimationOMP<PointXYZ, Normal> norm_est(16);

	norm_est.setSearchMethod (tree);
	norm_est.setRadiusSearch (2.5*radius);
	PointCloud<Normal> normals;
	
	FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
	int n1, n2, n3;
	fpfh_est.getNrSubdivisions(n1, n2, n3);
	std::cout << "fpfh sub num: " << n1 << " " << n2 << " " << n3 << std::endl;
	fpfh_est.setSearchMethod (tree);
	fpfh_est.setRadiusSearch (5.0*radius);
	PointCloud<FPFHSignature33> features_source, features_target;
	
	// Estimate the FPFH features for the source cloud
	norm_est.setInputCloud (cloud_source_ptr);
	norm_est.compute (normals);
	fpfh_est.setInputCloud (cloud_source_ptr);
	fpfh_est.setInputNormals (normals.makeShared ());
	fpfh_est.compute (features_source);
	
	// Estimate the FPFH features for the target cloud
	norm_est.setInputCloud (cloud_target_ptr);
	norm_est.compute (normals);
	fpfh_est.setInputCloud (cloud_target_ptr);
	fpfh_est.setInputNormals (normals.makeShared ());
	fpfh_est.compute (features_target);
	
	// Initialize Sample Consensus Initial Alignment (SAC-IA)
	SampleConsensusInitialAlignment<PointXYZ, PointXYZ, FPFHSignature33> reg;
	reg.setNumberOfSamples(3);
	/** \brief Get the minimum distances between samples, as set by the user */
	reg.setMinSampleDistance (10.0*radius); 
	/** \brief Get the maximum distance threshold between two correspondent points in source <-> target. If the 
	 * distance is larger than this threshold, the points will be ignored in the alignment process.
	 */
	reg.setMaxCorrespondenceDistance(1e5);
	//reg.setTransformationEpsilon(1);
	reg.setMaximumIterations(itnum);       
	reg.setCorrespondenceRandomness(5000);
	std::cout << "getMinSampleDistance: " << reg.getMinSampleDistance() << std::endl;
	std::cout << "getMaxCorrespondenceDistance: " << reg.getMaxCorrespondenceDistance() << std::endl;
	std::cout << "getNumberOfSamples: " << reg.getNumberOfSamples() << std::endl;
	std::cout << "getMaximumIterations: " << reg.getMaximumIterations() << std::endl;
	std::cout << "getCorrespondenceRandomness: " << reg.getCorrespondenceRandomness() << std::endl;
	
	reg.setInputSource (cloud_source_ptr);
	reg.setInputTarget (cloud_target_ptr);
	reg.setSourceFeatures (features_source.makeShared ());
	reg.setTargetFeatures (features_target.makeShared ());    
	// Register
	reg.align (cloud_sacia);

	for (int i = 0; i < 4; i++)
	    for (int j = 0; j < 4; j++)
	        transform(i,j) = reg.getFinalTransformation()(i,j);

	std::cout << "sacia fitnessscore: " << reg.getFitnessScore () << std::endl;
	return reg.getFitnessScore ();
    }
};
