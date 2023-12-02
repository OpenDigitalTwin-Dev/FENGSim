#include <pcl/point_cloud.h>
#include <pcl/features/feature.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>

#include <pcl/common/utils.h> // pcl::utils::ignore
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>


using namespace pcl;
using namespace pcl::io;

void test_normal_estimation ()
{
    PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(pcl::PointXYZ(1, 0, 0));
    cloud->push_back(pcl::PointXYZ(0, 1, 0));
    cloud->push_back(pcl::PointXYZ(0, 0, 1));

    // *********************************************
    // *********** details *************************
    // *********************************************
    Eigen::Vector4f centroid3;
    compute3DCentroid (*cloud, centroid3);
    std::cout << "***" << std::endl;
    std::cout << "centroid3: " << std::endl;
    std::cout << centroid3 << std::endl;
    Eigen::Matrix3f covariance_matrix;
    computeCovarianceMatrix (*cloud, centroid3, covariance_matrix);
    std::cout << "***" << std::endl;
    std::cout << "covariance_matrix: " << std::endl;
    std::cout << covariance_matrix << std::endl;
    Eigen::Vector4f plane_parameters;
    float curvature;
    solvePlaneParameters (covariance_matrix, centroid3, plane_parameters, curvature);
    std::cout << "***" << std::endl;
    std::cout << "eigenvalue and eigenvector: " << std::endl;
    std::cout << plane_parameters << std::endl;
    std::cout << curvature << std::endl << std::endl;    
    // solvePlaneParameters
    float nx, ny, nz;
    solvePlaneParameters (covariance_matrix, nx, ny, nz, curvature);
    std::cout << nx << " " << ny << " " << nz << std::endl;
    std::cout << curvature << std::endl;    

    // *********************************************
    // *********** normalestimation ****************
    // *********************************************
    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
    NormalEstimation<PointXYZ, Normal> normal_est;
    PointCloud<Normal> normals;
    normal_est.setInputCloud(cloud);
    normal_est.setSearchMethod(tree);

    std::cout << "***" << std::endl;
    std::cout << "normal estimation with radius 1.414214>sqrt(2)=1.414213562: " << std::endl;
    normal_est.setRadiusSearch(1.414214);
    normal_est.compute(normals);
    for (int i = 0; i < normals.size(); i++)
    {
        std::cout << normals[i] << std::endl;
    }
    std::cout << "normal estimation with radius 1.41421356<sqrt(2)=1.414213562: " << std::endl;
    normal_est.setRadiusSearch(1.41421356);
    normal_est.compute(normals);
    for (int i = 0; i < normals.size(); i++)
    {
        std::cout << normals[i] << std::endl;
    }
  
}
