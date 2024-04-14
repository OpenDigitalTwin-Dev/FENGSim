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
#include "odt_pcl.h"
#include "odt_pcl_tools.h"


using namespace pcl;
using namespace pcl::io;

PointCloud<PointXYZ> cloud_source, cloud_target, cloud_reg;
PointCloud<PointXYZRGBA> cloud_with_color;

void icp ()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>(500,1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the CloudIn data
  for (auto& point : *cloud_in)
  {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }

  *cloud_out = *cloud_in;

  for (auto& point : *cloud_out)
    point.x += 0.7f;

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  icp.setMaximumIterations(1000);
  icp.setEuclideanFitnessEpsilon(1e-5);


  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  export_pc_to_vtk(cloud_in,"test_icp_in.vtk");
  export_pc_to_vtk(cloud_out,"test_icp_out.vtk");
  export_pc_to_vtk(Final,"test_icp_final.vtk");
  // Check that we have sucessfully converged
  std::cout << icp.hasConverged() << std::endl;

  // Test that the fitness score is below acceptable threshold
  std::cout << icp.getFitnessScore() << std::endl;
}

int main ()
{
  icp ();
}
