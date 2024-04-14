/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

//#include <pcl/test/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/features/feature.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>

#include <pcl/common/utils.h> // pcl::utils::ignore
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

using namespace pcl;
using namespace pcl::io;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TEST (PCL, BaseFeature)
void TEST (PointCloud<PointXYZ> cloud)
{
  // compute3DCentroid (indices)
  Eigen::Vector4f centroid3;
  //compute3DCentroid (cloud, indices, centroid3);

  // compute3Dcentroid
  compute3DCentroid (cloud, centroid3);
  std::cout << "centroid3: " << std::endl;
  std::cout << centroid3 << std::endl;

  // computeNDCentroid (indices)
  //Eigen::VectorXf centroidn;
  //computeNDCentroid (cloud, indices, centroidn);

  // computeNDCentroid
  //computeNDCentroid (cloud, centroidn);
  
  // computeCovarianceMatrix (indices)
  Eigen::Matrix3f covariance_matrix;
  //computeCovarianceMatrix (cloud, indices, centroid3, covariance_matrix);

  // computeCovarianceMatrix
  computeCovarianceMatrix (cloud, centroid3, covariance_matrix);
  std::cout << "covariance_matrix: " << std::endl;
  std::cout << covariance_matrix << std::endl;
  
  // computeCovarianceMatrixNormalized (indices)
  //computeCovarianceMatrixNormalized (cloud, indices, centroid3, covariance_matrix);
  
  // computeCovarianceMatrixNormalized
  //computeCovarianceMatrixNormalized (cloud, centroid3, covariance_matrix);
  
  // solvePlaneParameters (Vector)
  Eigen::Vector4f plane_parameters;
  float curvature;
  solvePlaneParameters (covariance_matrix, centroid3, plane_parameters, curvature);
  std::cout << "eigenvalue and eigenvector: " << std::endl;
  std::cout << plane_parameters << std::endl;
  std::cout << curvature << std::endl;
  
  // solvePlaneParameters
  float nx, ny, nz;
  solvePlaneParameters (covariance_matrix, nx, ny, nz, curvature);
  std::cout << nx << " " << ny << " " << nz << std::endl;
  std::cout << curvature << std::endl;
 
}


#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

/* ---[ */
int main (int argc, char** argv) {

    PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(pcl::PointXYZ(1, 0, 0));
    cloud->push_back(pcl::PointXYZ(0, 1, 0));
    cloud->push_back(pcl::PointXYZ(0, 0, 1));
    //cloud->push_back(pcl::PointXYZ(0, 0, 2));
    TEST(*cloud);

    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
    NormalEstimation<PointXYZ, Normal> normal_est;
    PointCloud<Normal> normals;
    normal_est.setInputCloud (cloud);  
    normal_est.setRadiusSearch (1.5);  
    normal_est.setSearchMethod (tree);
    normal_est.compute (normals);

    std::cout << "normal estimation: " << std::endl;
    for (int i = 0; i < normals.size(); i++) {
        std::cout << normals[i] << std::endl;
    }



    
    
    return 0;

  
}
