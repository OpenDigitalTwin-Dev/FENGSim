#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/features/pfh_tools.h>

#include "odt_pcl.h"

double pi = 4*atan(1);
double radius = 0.6;

void feature ()
{
    float f1, f2, f3, f4;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "feature calculation" << std::endl;
    pcl::computePairFeatures(Eigen::Vector4f(0,0,0,0),
			     Eigen::Vector4f(1,2,3,0),
			     Eigen::Vector4f(1,0,0,0),
			     Eigen::Vector4f(4,5,6,0),f1,f2,f3,f4);
    std::cout << "results by computePairFeatures(): "<< f1 << " " << f2 << " " << f3 << " " << f4 << std::endl;
    std::cout << "compare the above results with pfh.py" << std::endl;
 
    // ############################################################
    // the following codes have been written in pfh.py
    Eigen::Vector4f p1(0,0,0,0);
    Eigen::Vector4f n1(1,2,3,0);
    Eigen::Vector4f p2(1,0,0,0);
    Eigen::Vector4f n2(4,5,6,0);

    Eigen::Vector4f dp2p1 = p2 - p1;
    dp2p1[3] = 0.0f;
    f4 = dp2p1.norm ();
    std::cout << "f4: " << f4 << std::endl;
    
    Eigen::Vector4f n1_copy = n1, n2_copy = n2;
    n1_copy[3] = n2_copy[3] = 0.0f;
    float angle1 = n1_copy.dot (dp2p1) / f4;
    float angle2 = n2_copy.dot (dp2p1) / f4;
    if (std::acos (std::fabs (angle1)) > std::acos (std::fabs (angle2))) {
        // switch p1 and p2
        n1_copy = n2;
	n2_copy = n1;
	n1_copy[3] = n2_copy[3] = 0.0f;
	dp2p1 *= (-1);
	f3 = -angle2;
    }
    else
        f3 = angle1;
    std::cout << "f3: " << f3 << std::endl;

    // Create a Darboux frame coordinate system u-v-w
    // u = n1; v = (p_idx - q_idx) x u / || (p_idx - q_idx) x u ||; w = u x v
    Eigen::Vector4f v = dp2p1.cross3 (n1_copy);
    v[3] = 0.0f;
    float v_norm = v.norm ();
    v /= v_norm;
    //std::cout << v << std::endl;

    
    Eigen::Vector4f w = n1_copy.cross3 (v);
    // Do not have to normalize w - it is a unit vector by construction
    
    v[3] = 0.0f;
    f2 = v.dot (n2_copy);
    std::cout << "f2: " << f2 << std::endl;
    w[3] = 0.0f;
    // Compute f1 = arctan (w * n2, u * n2) i.e. angle of n2 in the x=u, y=w coordinate system
    f1 = std::atan2 (w.dot (n2_copy), n1_copy.dot (n2_copy)); // @todo optimize this
    std::cout << "f1: " << f1 << std::endl;

}

void pfh () {
    int n = 10;
    float f1, f2, f3, f4;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "pfh to calculate point (0,0,0) feature" << std::endl;
    std::vector<int> index;
    std::vector<int> num;
    for (int i=0; i<n+1; i++) {
        for (int j=0; j<i; j++) {
	    Eigen::Vector4f p1(0+i*1.0/n,0,0,0);
	    Eigen::Vector4f n1(sin(i*pi/2/n), 0, cos(i*pi/2/n),0);
	    Eigen::Vector4f p2(0+j*1.0/n,0,0,0);
	    Eigen::Vector4f n2(sin(j*pi/2/n), 0, cos(j*pi/2/n),0);

	    Eigen::Vector4f p(0,0,0,0);
	    Eigen::Vector4f dp1p = p1 - p;
	    dp1p[3] = 0.0f;
	    Eigen::Vector4f dp2p = p2 - p;
	    dp2p[3] = 0.0f;

	    if (dp1p.norm ()>=0.5 || dp2p.norm ()>=0.5) continue;
 
	    pcl::computePairFeatures(p1,n1,p2,n2,f1,f2,f3,f4); 
	    double d_pi_ = 1.0f / (2.0f * static_cast<float> (M_PI));
	    int id1 = std::floor (5.0 * ((f1 + M_PI) * d_pi_));
	    int id2 = std::floor (5.0 * ((f2 + 1.0) * 0.5));
	    int id3 = std::floor (5.0 * ((f3 + 1.0) * 0.5));
	    int id = id1 + 5*id2 + 5*5*id3;
	    
	    bool stop = false;
	    for (int k=0; k<index.size(); k++) {
	        if (index[k]==id) {
		    num[k] = num[k] + 1;
		    stop = true;
		}
	    }
	    if (stop) {
	        continue;
	    }
	    else {
	        index.push_back(id);
		num.push_back(1);
	    }
	}
    }
    for (int i=0; i<index.size(); i++)
        std::cout << index[i] << std::endl;
    double total = 0;
    for (int i=0; i<num.size(); i++)
        total += num[i];
    for (int i=0; i<num.size(); i++)
        std::cout << double(num[i]) / total << std::endl;
}

void spfh (std::vector<double>& index, Eigen::Vector4f p1=Eigen::Vector4f(0+0*1.0/10,0,0,0),
	   Eigen::Vector4f n1=Eigen::Vector4f(sin(0*pi/2/10), 0, cos(0*pi/2/10),0))
{
    int n = 10;
    float f1, f2, f3, f4;
    int indices = 0;
    for (int i=0; i<n+1; i++) {
        Eigen::Vector4f p2(0+i*1.0/n,0,0,0);
	Eigen::Vector4f n2(sin(i*pi/2/n), 0, cos(i*pi/2/n),0);
	Eigen::Vector4f dp1p2 = p2 - p1;
	dp1p2[3] = 0.0f;
	if (dp1p2.norm ()>=radius) continue;
	indices++;
    }	
    for (int i=0; i<n+1; i++) {
        Eigen::Vector4f p2(0+i*1.0/n,0,0,0);
	Eigen::Vector4f n2(sin(i*pi/2/n), 0, cos(i*pi/2/n),0);
	Eigen::Vector4f dp1p2 = p2 - p1;
	dp1p2[3] = 0.0f;
	if (dp1p2.norm ()>=radius || dp1p2.norm ()==0) continue;

	pcl::computePairFeatures(p1,n1,p2,n2,f1,f2,f3,f4);
	//std::cout << f1 << " " << f2 << " " << f3 << std::endl;
	double d_pi_ = 1.0f / (2.0f * static_cast<float> (M_PI));
	int id1 = std::floor (11.0 * ((f1 + M_PI) * d_pi_));
	int id2 = std::floor (11.0 * ((f2 + 1.0) * 0.5));
	int id3 = std::floor (11.0 * ((f3 + 1.0) * 0.5));

	float hist_incr = 100.0f / static_cast<float>(indices-1);
	//std::cout << "hist_incr: " << hist_incr << std::endl;
	//std::cout << id1 << " "  << id2 << " "  << id3 << ": " << hist_incr << std::endl;
	index[id1] = index[id1] + hist_incr;
	index[11+id2] = index[11+id2] + hist_incr;
	index[22+id3] = index[22+id3] + hist_incr; 
    }
    for (int i=0; i<3; i++) {
        double total = 0;
	for (int j=0; j<11; j++) {
	    total += index[i*11+j];
	}
	if (total==0) continue;
	for (int j=0; j<11; j++)
	    index[i*11+j] =  double(index[i*11+j]) / total * 100; 
    }
}

void fpfh () {
    int n = 10;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "fpfh to calculate point (0,0,0) feature" << std::endl;
    std::vector<double> index1(33,0);
    Eigen::Vector4f p1(0+0*1.0/n,0,0,0);
    Eigen::Vector4f n1(sin(0*pi/2/n), 0, cos(0*pi/2/n),0);
    for (int i=0; i<n+1; i++) {
        Eigen::Vector4f p2(0+i*1.0/n,0,0,0);
	Eigen::Vector4f n2(sin(i*pi/2/n), 0, cos(i*pi/2/n),0);
	Eigen::Vector4f dp1p2 = p1 - p2;
	dp1p2[3] = 0.0f;
	if (dp1p2.norm ()>=radius || dp1p2.norm ()==0) continue;
	std::vector<double> index2(33,0);
        spfh(index2,p2,n2);
	double w = 1.0 / dp1p2.norm () / dp1p2.norm ();
	for (int j=0; j<33; j++) {
	    index1[j] += w * index2[j];
	}
    }
    for (int i=0; i<3; i++) {
        double total = 0;
	for (int j=0; j<11; j++) {
	    total += index1[i*11+j];
	}
	for (int j=0; j<11; j++)
	    std::cout << index1[i*11+j] * (100.0 / total) << " ";
	std::cout << std::endl;
    }
}

void test_pfh_estimation () {
    // ############################################################
    feature();

    // ############################################################
    pfh();

    // ############################################################
    std::cout << "------------------------------------" << std::endl;
    std::cout << "results by PFHEstimation()" << std::endl;
    int n = 10;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; i<n+1; i++) {
        cloud->push_back(pcl::PointXYZ(0+i*1.0/n, 0, 0));
    }
    
    pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);
    for (int i=0; i<n+1; i++) {
        cloud_with_normals->push_back(pcl::Normal(sin(i*pi/2/n), 0, cos(i*pi/2/n)));
    }

    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh_estimation;
    pfh_estimation.setInputCloud (cloud);
    pfh_estimation.setInputNormals (cloud_with_normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pfh_estimation.setSearchMethod (tree);    
    pfh_estimation.setRadiusSearch (0.4);
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_features (new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_estimation.compute (*pfh_features);
    std::cout << "output size (): " << pfh_features->size () << std::endl;
    
    // Display and retrieve the shape context descriptor vector for the 0th point.
    pcl::PFHSignature125 descriptor = (*pfh_features)[0];
    std::cout << descriptor << std::endl;
    pcl::visualization::PCLPlotter plotter("pfh");
    plotter.addFeatureHistogram(*pfh_features,125);
    //plotter.plot();

    // ############################################################
    fpfh();
    
    // ############################################################
    std::cout << "------------------------------------" << std::endl;
    std::cout << "results by FPFHEstimation()" << std::endl;
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setInputCloud (cloud);
    fpfh_estimation.setInputNormals (cloud_with_normals);
    fpfh_estimation.setSearchMethod (tree);    
    fpfh_estimation.setRadiusSearch (radius);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features (new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh_estimation.compute (*fpfh_features);
    std::cout << "output size (): " << fpfh_features->size () << std::endl;

    pcl::FPFHSignature33 descriptor33 = (*fpfh_features)[0];
    std::cout << descriptor33 << std::endl;
    plotter.addFeatureHistogram(*fpfh_features,(n+1)*33);
    //plotter.plot();

}
