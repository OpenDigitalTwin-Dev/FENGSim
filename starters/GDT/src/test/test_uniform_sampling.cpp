#include <pcl/common/generate.h>
#include <pcl/common/random.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/point_types.h>

#include "odt_pcl.h"
#include "odt_pcl_tools.h"

void test_uniform_sampling () {
    using namespace pcl::common;

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>);
    /*
    const int SEED = 1234;
    CloudGenerator<pcl::PointXYZ, UniformGenerator<float>> generator;
    UniformGenerator<float>::Parameters x_params(0, 1, SEED + 1);
    generator.setParametersForX(x_params);
    UniformGenerator<float>::Parameters y_params(0, 1, SEED + 2);
    generator.setParametersForY(y_params);
    UniformGenerator<float>::Parameters z_params(0, 1, SEED + 3);
    generator.setParametersForZ(z_params);
    generator.fill(100, 100, *xyz);
    */
    xyz->push_back(pcl::PointXYZ(0, 0, 0));
    xyz->push_back(pcl::PointXYZ(1, 0, 0));
    xyz->push_back(pcl::PointXYZ(1, 1, 0));
    xyz->push_back(pcl::PointXYZ(0, 1, 0));
    xyz->push_back(pcl::PointXYZ(0, 0, 1));
    xyz->push_back(pcl::PointXYZ(1, 0, 1));
    xyz->push_back(pcl::PointXYZ(1, 1, 1));
    xyz->push_back(pcl::PointXYZ(0, 1, 1));  
    // The generated cloud points are distributed in the unit cube. By using 0.1 sized
    // voxels for sampling, we divide each side of the cube into 1 / 0.1 = 10 cells, in
    // total 10^3 = 1000 cells. Since we generated a large amount of points, we can be
    // sure that each cell has at least one point. As a result, we expect 1000 points in
    // the output cloud and the rest in removed indices.

    pcl::UniformSampling<pcl::PointXYZ> us(true); // extract removed indices
    us.setInputCloud(xyz);
    double radius = 0.9999;
    us.setRadiusSearch(radius);
    pcl::PointCloud<pcl::PointXYZ> output;
    us.filter(output);
    std::cout << "radius: " << 0.9999 << std::endl;
    std::cout << "xyz number: " << (*xyz).size() << std::endl;
    std::cout << "uniform sampling number: " << output.size() << std::endl;

    for (int i = 0; i < output.size(); i++)
    {
        std::cout << "(" << output.points[i].x << ", ";
	std::cout << output.points[i].y << ", ";
	std::cout << output.points[i].z << ")" << std::endl;
    }

    radius = 1.0001;
    us.setRadiusSearch(radius);
    us.filter(output);
    std::cout << "radius: " << 1.0001 << std::endl;
    std::cout << "xyz number: " << (*xyz).size() << std::endl;
    std::cout << "uniform sampling number: " << output.size() << std::endl;

    for (int i = 0; i < output.size(); i++)
    {
        std::cout << "(" << output.points[i].x << ", ";
	std::cout << output.points[i].y << ", ";
	std::cout << output.points[i].z << ")" << std::endl;
    }
    
}


