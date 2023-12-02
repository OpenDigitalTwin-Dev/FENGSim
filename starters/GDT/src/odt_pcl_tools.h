#ifndef _ODT_PCL_TOOLS_H_
#define _ODT_PCL_TOOLS_H_


#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <fstream>


void export_pc_to_vtk (pcl::PointCloud<pcl::PointXYZ> pc, std::string filename);

void export_pc_to_vtk (pcl::PointCloud<pcl::PointXYZ>::Ptr pc, std::string filename);

void export_matrix (Eigen::Matrix4f transform, std::string filename);

void import_pc (std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr pc);

void import_pc (std::string filename, pcl::PointCloud<pcl::PointXYZ> &pc);

void import_mesh2pc (std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr pc);

void import_mesh2pc (std::string filename, pcl::PointCloud<pcl::PointXYZ> &pc);

void source_mesh_tran (Eigen::Matrix4f transform_icp, Eigen::Matrix4f transform_sacia);

void voxel_grid (pcl::PointCloud<pcl::PointXYZ>& pc, double t, std::string filename);

#endif
