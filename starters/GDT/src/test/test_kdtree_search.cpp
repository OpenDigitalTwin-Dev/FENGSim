#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <iostream>
#include <vector>

#include "odt_pcl.h"
#include "odt_pcl_tools.h"

using namespace pcl;

void test_kdtree_search () {  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(pcl::PointXYZ(30, 40, 0));
    cloud->push_back(pcl::PointXYZ(5, 25, 0));
    cloud->push_back(pcl::PointXYZ(10, 12, 0));
    cloud->push_back(pcl::PointXYZ(70, 70, 0));
    cloud->push_back(pcl::PointXYZ(50, 30, 0));
    cloud->push_back(pcl::PointXYZ(35, 45, 0));
    
    search::KdTree<PointXYZ>::Ptr kdtree (new search::KdTree<PointXYZ>);
    kdtree->setInputCloud (cloud);
    
    pcl::PointXYZ searchPoint(30,40,0);
    std::cout << "***" << std::endl;
    for (int i=0; i<(*cloud).size(); i++)
    {
        std::cout << (*cloud)[i].x << " " << (*cloud)[i].y << ": "
		  << pow((*cloud)[i].x-30,2) + pow((*cloud)[i].y-40,2) << std::endl;
    }
    std::cout << "***" << std::endl;
    
    // K nearest neighbor search
    int K = 3;
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    
    std::cout << "K nearest neighbor search at (" << searchPoint.x 
	      << " " << searchPoint.y 
	      << " " << searchPoint.z
	      << ") with K=" << K << std::endl;
    
    if ( kdtree->nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
        for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i) {
	    std::cout << (*cloud)[pointIdxNKNSearch[i]].x 
		      << " " << (*cloud)[pointIdxNKNSearch[i]].y 
		      << " " << (*cloud)[pointIdxNKNSearch[i]].z 
		      << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
	}
    }
    
    // Neighbors within radius search
    
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    
    float radius = sqrt(850);
    
    std::cout << "Neighbors within radius search at (" << searchPoint.x 
	      << " " << searchPoint.y 
	      << " " << searchPoint.z
	      << ") with radius=sqrt(850)=29.154759474 (" << radius << ")"<< std::endl;
    

    if ( kdtree->radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
        for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i) {
	    std::cout << (*cloud)[pointIdxRadiusSearch[i]].x 
		      << " " << (*cloud)[pointIdxRadiusSearch[i]].y 
		      << " " << (*cloud)[pointIdxRadiusSearch[i]].z 
		      << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
    }
    
}

