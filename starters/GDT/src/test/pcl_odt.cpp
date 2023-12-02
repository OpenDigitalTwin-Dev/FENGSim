/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2019-, Open Perception, Inc.
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
 */


#include <pcl/common/generate.h>
#include <pcl/common/random.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/point_types.h>
#include "fengsim_pcl.h"

void test_uniform_sampling () {
    using namespace pcl::common;
    const int SEED = 1234;
    CloudGenerator<pcl::PointXYZ, UniformGenerator<float>> generator;
    UniformGenerator<float>::Parameters x_params(0, 1, SEED + 1);
    generator.setParametersForX(x_params);
    UniformGenerator<float>::Parameters y_params(0, 1, SEED + 2);
    generator.setParametersForY(y_params);
    UniformGenerator<float>::Parameters z_params(0, 1, SEED + 3);
    generator.setParametersForZ(z_params);
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>);
    //generator.fill(100, 100, *xyz);

    
    xyz->push_back(pcl::PointXYZ(0, 0, 0));
    xyz->push_back(pcl::PointXYZ(1, 0, 0));
    xyz->push_back(pcl::PointXYZ(1, 1, 0));
    xyz->push_back(pcl::PointXYZ(0, 1, 0));
    //xyz->push_back(pcl::PointXYZ(0.9, 0, 0));
    //xyz->push_back(pcl::PointXYZ(1.1, 0, 0));
    
    
    // The generated cloud points are distributed in the unit cube. By using 0.1 sized
    // voxels for sampling, we divide each side of the cube into 1 / 0.1 = 10 cells, in
    // total 10^3 = 1000 cells. Since we generated a large amount of points, we can be
    // sure that each cell has at least one point. As a result, we expect 1000 points in
    // the output cloud and the rest in removed indices.

    pcl::UniformSampling<pcl::PointXYZ> us(true); // extract removed indices
    us.setInputCloud(xyz);
    double radius = 0.1;
    us.setRadiusSearch(radius);
    pcl::PointCloud<pcl::PointXYZ> output;
    us.filter(output);
    std::cout << "xyz number: " << (*xyz).size() << std::endl;
    std::cout << "uniform sampling number: " << output.size() << std::endl;

    export_pc_to_vtk(*xyz, "./origin_point_cloud.vtk");
    voxel_grid(*xyz, radius, "./voxel_grid.vtk");
    export_pc_to_vtk(output, "./uniform_sampling_point_cloud.vtk");

    return;
    for (int i = 0; i < output.size(); i++) {
        std::cout << "(" << output.points[i].x << ", ";
	std::cout << output.points[i].y << ", ";
	std::cout << output.points[i].z << ")" << std::endl;
    }
}
/*
int main(int argc, char** argv) {
    test();
    return 0;
}
*/
