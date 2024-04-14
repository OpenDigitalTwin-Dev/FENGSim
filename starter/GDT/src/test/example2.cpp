#include "example2.h"

#include <fstream>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/sample_consensus_prerejective.h>


#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
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
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/pyramid_feature_matching.h>
#include <pcl/features/ppf.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/sample_consensus_prerejective.h>
// We need Histogram<2> to function, so we'll explicitely add kdtree_flann.hpp here
#include <pcl/kdtree/impl/kdtree_flann.hpp>



using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace pcl::registration;




////////////////////////////////////////////////////////////////////////////////
void
estimateKeypoints (const PointCloud<PointXYZ>::Ptr &src, 
                   const PointCloud<PointXYZ>::Ptr &tgt,
                   PointCloud<PointXYZ> &keypoints_src,
                   PointCloud<PointXYZ> &keypoints_tgt)
{
  // Get an uniform grid of keypoints
  UniformSampling<PointXYZ> uniform;
  uniform.setRadiusSearch (0.25);  // 1m

  uniform.setInputCloud (src);
  uniform.filter (keypoints_src);

  uniform.setInputCloud (tgt);
  uniform.filter (keypoints_tgt);

  return;
}

////////////////////////////////////////////////////////////////////////////////
void
estimateNormals (const PointCloud<PointXYZ>::Ptr &src, 
                 const PointCloud<PointXYZ>::Ptr &tgt,
                 PointCloud<Normal> &normals_src,
                 PointCloud<Normal> &normals_tgt)
{
    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
    NormalEstimation<PointXYZ, Normal> normal_est;
    normal_est.setInputCloud (src);  
    normal_est.setRadiusSearch (0.1);  // 50cm
    normal_est.setSearchMethod (tree);
    
    //    normal_est.setViewPoint(1000,1000,1000);
    normal_est.compute (normals_src);
    
    normal_est.setInputCloud (tgt);
    normal_est.compute (normals_tgt);

    return;
}

////////////////////////////////////////////////////////////////////////////////
void
estimateFPFH (const PointCloud<PointXYZ>::Ptr &src, 
              const PointCloud<PointXYZ>::Ptr &tgt,
              const PointCloud<Normal>::Ptr &normals_src,
              const PointCloud<Normal>::Ptr &normals_tgt,
              const PointCloud<PointXYZ>::Ptr &keypoints_src,
              const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
              PointCloud<FPFHSignature33> &fpfhs_src,
              PointCloud<FPFHSignature33> &fpfhs_tgt)
{

    search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ>);
  
    FPFHEstimation<PointXYZ, Normal, FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud (keypoints_src);
    fpfh_est.setInputNormals (normals_src);



  
    //pcl::search::KdTree<PointXYZ>::Ptr tree (new pcl::search::KdTree<PointXYZ>);
    fpfh_est.setSearchMethod (tree);
    fpfh_est.setRadiusSearch (0.1); // 1m
    
    
    fpfh_est.setSearchSurface (src);
    fpfh_est.compute (fpfhs_src);


  
    fpfh_est.setInputCloud (keypoints_tgt);
    fpfh_est.setInputNormals (normals_tgt);
    fpfh_est.setSearchSurface (tgt);
    fpfh_est.compute (fpfhs_tgt);




}

////////////////////////////////////////////////////////////////////////////////
void
findCorrespondences (const PointCloud<FPFHSignature33>::Ptr &fpfhs_src,
                     const PointCloud<FPFHSignature33>::Ptr &fpfhs_tgt,
                     Correspondences &all_correspondences)
{
    CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> est;
    //est.setInputCloud(fpfhs_src);
    est.setInputSource(fpfhs_src);
    est.setInputTarget(fpfhs_tgt);
    est.determineReciprocalCorrespondences(all_correspondences);
    //est.determineCorrespondences(all_correspondences,10);
}

////////////////////////////////////////////////////////////////////////////////
void
rejectBadCorrespondences (const CorrespondencesPtr &all_correspondences,
                          const PointCloud<PointXYZ>::Ptr &keypoints_src,
                          const PointCloud<PointXYZ>::Ptr &keypoints_tgt,
                          Correspondences &remaining_correspondences)
{
    CorrespondenceRejectorDistance rej;
    rej.setInputSource<PointXYZ> (keypoints_src);
    rej.setInputTarget<PointXYZ> (keypoints_tgt);
    
    rej.setMaximumDistance (5);    // 1m
    rej.setInputCorrespondences (all_correspondences);
    rej.getCorrespondences (remaining_correspondences);
}


////////////////////////////////////////////////////////////////////////////////
void
computeTransformation (const PointCloud<PointXYZ>::Ptr &src, 
                       const PointCloud<PointXYZ>::Ptr &tgt,
                       Eigen::Matrix4f &transform)
{
  
    // Get an uniform grid of keypoints
    PointCloud<PointXYZ>::Ptr keypoints_src (new PointCloud<PointXYZ>), 
	keypoints_tgt (new PointCloud<PointXYZ>);
    
    estimateKeypoints (src, tgt, *keypoints_src, *keypoints_tgt);
    print_info ("Found %zu and %zu keypoints for the source and target datasets.\n", static_cast<std::size_t>(keypoints_src->size ()), static_cast<std::size_t>(keypoints_tgt->size ()));
    
    // Compute normals for all points keypoint
    PointCloud<Normal>::Ptr normals_src (new PointCloud<Normal>), 
	normals_tgt (new PointCloud<Normal>);
    estimateNormals (src, tgt, *normals_src, *normals_tgt);
    print_info ("Estimated %zu and %zu normals for the source and target datasets.\n", static_cast<std::size_t>(normals_src->size ()), static_cast<std::size_t>(normals_tgt->size ()));
    
    // Compute FPFH features at each keypoint
    PointCloud<FPFHSignature33>::Ptr fpfhs_src (new PointCloud<FPFHSignature33>), 
	fpfhs_tgt (new PointCloud<FPFHSignature33>);
    estimateFPFH (src, tgt, normals_src, normals_tgt, keypoints_src, keypoints_tgt, *fpfhs_src, *fpfhs_tgt);
    
    // Copy the data and save it to disk
    /*  PointCloud<PointNormal> s, t;
	copyPointCloud (*keypoints_src, s);
	copyPointCloud (normals_src, s);
	copyPointCloud (*keypoints_tgt, t);
	copyPointCloud (normals_tgt, t);*/
    
    // Find correspondences between keypoints in FPFH space
    CorrespondencesPtr all_correspondences (new Correspondences), 
	good_correspondences (new Correspondences);
    findCorrespondences (fpfhs_src, fpfhs_tgt, *all_correspondences);

    
    //std::cout << std::endl << "all_correspondences: " << all_correspondences->size() << std::endl;
  

    // Reject correspondences based on their XYZ distance
    rejectBadCorrespondences (all_correspondences, keypoints_src, keypoints_tgt, *good_correspondences);
    
    //for (int i = 0; i < good_correspondences->size (); ++i)
    //	std::cerr << good_correspondences->at (i) << std::endl;
    //std::cout << std::endl;
    // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
    //TransformationEstimationSVD<PointXYZ, PointXYZ> trans_est;
    //trans_est.estimateRigidTransformation (*keypoints_src, *keypoints_tgt, *good_correspondences, transform);
    
    







    SampleConsensusInitialAlignment<PointXYZ, PointXYZ, FPFHSignature33> scia;
    scia.setInputSource(keypoints_src);
    scia.setInputTarget(keypoints_tgt);
    scia.setSourceFeatures(fpfhs_src);
    scia.setTargetFeatures(fpfhs_tgt);
    //scia.setMinSampleDistance(1);
    //scia.setNumberOfSamples(2);
    //scia.setCorrespondenceRandomness(20);
    PointCloud<PointXYZ>::Ptr sac_result(new PointCloud<PointXYZ>);
    scia.align(*sac_result);
    std::cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << std::endl;
    Eigen::Matrix4f sac_trans;
    sac_trans = scia.getFinalTransformation();
    transform = scia.getFinalTransformation();
    std::cout << sac_trans << std::endl;
    //pcl::io::savePCDFileASCII("junjie_transformed_sac.pcd", *sac_result);
    clock_t sac_time = clock();



    












    IterativeClosestPoint<PointXYZ, PointXYZ, double> icp;
    PointCloud<PointXYZ> cloud_src2;
    transformPointCloud (*src, cloud_src2, transform);
    icp.setInputSource(cloud_src2.makeShared());
    icp.setInputTarget(tgt);
    
    //std::cout << "icp parameter: " << p1 << " " << p2 << " " << p3 << std::endl;
    
    
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-5);
    icp.setEuclideanFitnessEpsilon(1e-5);
    
    
    //icp.setMaxCorrespondenceDistance(1e-5);
    //icp.setRANSACOutlierRejectionThreshold(1e-5);
    
    pcl::PointCloud<pcl::PointXYZ> output_icp;
    icp.align(output_icp);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
	icp.getFitnessScore() << " fitness epsilon: " << icp.getEuclideanFitnessEpsilon() <<
	" transform epsilon: " << icp.getTransformationEpsilon() << std::endl;

    
    std::cout << icp.getFinalTransformation() << std::endl;
    Eigen::Matrix4f transform2;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
	transform2(i,j) = icp.getFinalTransformation()(i,j);
    Eigen::Matrix4f inverse_transform = transform.inverse();
    Eigen::Matrix4f inverse_transform2 = transform2.inverse();
    PointCloud<PointXYZ> tgt2;
    PointCloud<PointXYZ> tgt3;
    transformPointCloud (*tgt, tgt2, inverse_transform2);
    transformPointCloud (tgt2, tgt3, inverse_transform);



    


    std::ifstream is("./data/meas/fengsim_meas_scene_mesh_points.vtk");
    std::ofstream out("./data/meas/fengsim_meas_scene_mesh_points_trans.vtk");
    const int len = 256;
    char L[len];
    while (is.getline(L,len)) {
        double z[3];
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
	Eigen::Vector4f v1(z[0], z[1], z[2], 1);
	Eigen::Vector4f v2 = inverse_transform * inverse_transform2 * v1;
	out << v2[0] << " "
	    << v2[1] << " "
	    << v2[2] << std::endl;
    }
    is.close();
    out.close();


    
    return;    

    //dock->ui->textEdit->append(QString("has converged:") + QString::number(icp.hasConverged()) + QString("\r\n"));
    //dock->ui->textEdit->append(QString("score:") + QString::number(icp.getFitnessScore()) + QString("\r\n"));
    //dock->ui->textEdit->append(QString("fitness epsilon: ") + QString::number(icp.getFitnessScore()) + QString("\r\n"));
    
    //std::ofstream out;
    out.open("./data/meas/output_icp.vtk");
    
    for (int i = 0; i < output_icp.size(); i++)
	{
	    out << std::setprecision(16)
		//out
		<< (output_icp.points)[i].x << " "
		<< (output_icp.points)[i].y << " "
		<< (output_icp.points)[i].z << std::endl;
	    
	}
    out.close();
    //std::cout <<"icp final tol: " << tol1 << std::endl;
    
    
    
    
    out.open("./data/meas/output_icp2.vtk");
    
    for (int i = 0; i < tgt3.size(); i++)
	{
	    out << std::setprecision(16)
		//out
		<< (tgt3.points)[i].x << " "
		<< (tgt3.points)[i].y << " "
		<< (tgt3.points)[i].z << std::endl;
	    
	}
    out.close();
    






    



  
}

/* ---[ */


int
//registration_gdt ()
main ()
{

    PointCloud<PointXYZ>::Ptr src, tgt;
    pcl::PointCloud<pcl::PointXYZ> cloud_src;
    pcl::PointCloud<pcl::PointXYZ> cloud_tgt;
    /*    bool t = false;
    if (t) {
	std::ifstream is("./fengsim_meas_scene.vtk");
	const int len = 256;
	char L[len];
	for (int i = 0; i < 5; i++)
	    is.getline(L,len);
	int n;
	int d = sscanf(L,"%*s %d %*s", &n);
	std::cout << d << " " << n << std::endl;
	

	cloud_src.width    = n;
	cloud_src.height   = 1;
	cloud_src.is_dense = false;
	cloud_src.points.resize (cloud_src.width * cloud_src.height);
	for (int i = 0; i < n; i++) {
	    is.getline(L,len);
	    double z[3];
	    sscanf(L,"%lf %lf %lf", z, z+1, z+2);
	    cloud_src.points[i].x = z[0];
	    cloud_src.points[i].y = z[1];
	    cloud_src.points[i].z = z[2];
	}
	is.close();
	pcl::io::savePCDFile("cloud_src.pcd", cloud_src);
	
	

	is.open("./fengsim_meas_model.vtk");
	for (int i = 0; i < 5; i++)
	    is.getline(L,len);
	d = sscanf(L,"%*s %d %*s", &n);
	std::cout << n << std::endl;
	

	cloud_tgt.width    = n;
	cloud_tgt.height   = 1;
	cloud_tgt.is_dense = false;
	cloud_tgt.points.resize (cloud_tgt.width * cloud_tgt.height);
	for (int i = 0; i < n; i++) {
	    is.getline(L,len);
	    double z[3];
	    sscanf(L,"%lf %lf %lf", z, z+1, z+2);
	    cloud_tgt.points[i].x = z[0];
	    cloud_tgt.points[i].y = z[1];
	    cloud_tgt.points[i].z = z[2];
	}
	is.close();
	pcl::io::savePCDFile("cloud_tgt.pcd", cloud_tgt);
    }
    else {*/


    std::ifstream is("./data/meas/fengsim_meas_model.vtk");
    const int len = 256;
    char L[len];
    int n = 0;
    while (is.getline(L,len)) {
	n++;
    }
    is.close();
    
    //pcl::PointCloud<pcl::PointXYZ> cloud_src;
    cloud_src.width    = n;
    cloud_src.height   = 1;
    cloud_src.is_dense = false;
    cloud_src.points.resize (cloud_src.width * cloud_src.height);
    is.open("./data/meas/fengsim_meas_model.vtk");
    int i = 0;
    while (is.getline(L,len)) {
	double z[3];
	sscanf(L,"%lf %lf %lf", z, z+1, z+2);
	cloud_src.points[i].x = z[0];
	cloud_src.points[i].y = z[1];
	cloud_src.points[i].z = z[2];
	i++;
    }
    is.close();
    //pcl::io::savePCDFile("cloud_src.pcd", cloud_src);
	
	

    is.open("./data/meas/fengsim_meas_scene.vtk");
    n = 0;
    while (is.getline(L,len)) {
	n++;
    }
    is.close();
    std::cout << n << std::endl;
	
    //pcl::PointCloud<pcl::PointXYZ> cloud_tgt;
    cloud_tgt.width    = n;
    cloud_tgt.height   = 1;
    cloud_tgt.is_dense = false;
    cloud_tgt.points.resize (cloud_tgt.width * cloud_tgt.height);
    is.open("./data/meas/fengsim_meas_scene.vtk");
    i = 0;
    while (is.getline(L,len)) {
	double z[3];
	sscanf(L,"%lf %lf %lf", z, z+1, z+2);
	cloud_tgt.points[i].x = z[0];
	cloud_tgt.points[i].y = z[1];
	cloud_tgt.points[i].z = z[2];
	i++;
    }
    is.close();
    //pcl::io::savePCDFile("cloud_tgt.pcd", cloud_tgt);
    //}
















  
    // Parse the command line arguments for .pcd files
    //std::vector<int> p_file_indices;
    //p_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
    /*    if (p_file_indices.size () != 2)
    {
	print_error ("Need one input source PCD file and one input target PCD file to continue.\n");
	print_error ("Example: %s source.pcd target.pcd\n", argv[0]);
	return (-1);
      }
    */
    // Load the files
    //print_info ("Loading %s as source and %s as target...\n", argv[p_file_indices[0]], argv[p_file_indices[1]]);
    src.reset (new PointCloud<PointXYZ>(cloud_src));
    tgt.reset (new PointCloud<PointXYZ>(cloud_tgt));
    
    //  if (loadPCDFile (argv[p_file_indices[0]], *src) == -1 || loadPCDFile (argv[p_file_indices[1]], *tgt) == -1)
    //{
    //print_error ("Error reading the input files!\n");
    //return (-1);
    //}
    








    /*
    //SAC配准
    
    */












    // Compute the best transformtion
    Eigen::Matrix4f transform;
    computeTransformation (src, tgt, transform);
    
    
    return 0;
    
    
    std::cerr << transform << std::endl;
    // Transform the data and write it to disk
    PointCloud<PointXYZ> output_init;
    transformPointCloud (*src, output_init, transform);
    savePCDFileBinary ("source_transformed.pcd", output_init);
    

    std::ofstream out("./output_init.vtk");
    for (int i = 0; i < output_init.size(); i++) {
      out << output_init.points[i].x << " ";
      out << output_init.points[i].y << " ";
      out << output_init.points[i].z << std::endl;
    }

  
}
/* ]--- */









/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>             // 法线
#include <pcl/features/fpfh_omp.h>
#include <pcl/visualization/pcl_visualizer.h>   // 可视化
#include <pcl/visualization/pcl_plotter.h>

#include <iostream>
#include <chrono>

using namespace std;

boost::mutex cloud_mutex;
pcl::visualization::PCLPlotter plotter;
//pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh_omp;
pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

// structure used to pass arguments to the callback function
struct callback_args {
    pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d;
    pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

// callback function
void pp_callback(const pcl::visualization::PointPickingEvent& event, void* args)
{
    plotter.clearPlots();
    struct callback_args* data = (struct callback_args *)args;
    if (event.getPointIndex() == -1)
        return;
    pcl::PointXYZ current_point;
    event.getPoint(current_point.x, current_point.y, current_point.z);
    data->clicked_points_3d->points.clear();
    data->clicked_points_3d->points.push_back(current_point);

    // Draw clicked points in red:
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(data->clicked_points_3d, 255, 0, 0);
    data->viewerPtr->removePointCloud("clicked_points");
    data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
    data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
    std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;

    int num = event.getPointIndex();
    plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, "fpfh", num);
    plotter.plot();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf(" -- Usage: %s <pointcloud file>\n", argv[0]);
        return -1;
    }

    bool display = true;
    bool downSampling = false;

    // load pcd/ply point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>()); // 模型点云
    if (pcl::io::loadPCDFile(argv[1], *model) < 0) {
        std::cerr << "Error loading model cloud." << std::endl;
        return -1;
    }
    std::cout << "Cloud size: " << model->points.size() << std::endl;

//    cloud_mutex.lock();// for not overwriting the point cloud
    if (downSampling) {
        // create the filtering object
        std::cout << "Number of points before downSampling: " << model->points.size() << std::endl;
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(model);
        sor.setLeafSize(0.01, 0.01, 0.01);
        sor.filter(*model);
        std::cout << "Number of points after downSampling: " << model->points.size() << std::endl;
    }

    //  Normal estimation
    auto t1 = chrono::steady_clock::now();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(model);
    ne.setSearchMethod(tree);
    ne.setKSearch(10);
//    ne.setRadiusSearch(0.03);
    ne.compute(*normals);
    auto t2 = chrono::steady_clock::now();
    auto dt = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    cout << "Time cost of Normal estimation: " << dt << endl;

    // fpfh or fpfh_omp
    fpfh_omp.setInputCloud(model);
    fpfh_omp.setInputNormals(normals);
    fpfh_omp.setSearchMethod(tree);
    fpfh_omp.setNumberOfThreads(8);
    fpfh_omp.setRadiusSearch(0.05);
    fpfh_omp.compute(*fpfhs);
//    fpfh.setInputCloud(model);
//    fpfh.setInputNormals(normals);
//    fpfh.setSearchMethod(tree);
//    fpfh.setRadiusSearch(0.05);
//    fpfh.compute(*fpfhs);
    t1 = chrono::steady_clock::now();
    dt = chrono::duration_cast<chrono::duration<double> >(t1 - t2).count();
    cout << "Time cost of FPFH estimation: " << dt << endl;

    pcl::FPFHSignature33 descriptor;
    for (int i=0; i<10; ++i) {
        int index = i + rand() % model->points.size();
        descriptor = fpfhs->points[index];
        std::cout << " -- fpfh for point "<< index << ":\n" << descriptor << std::endl;
    }

    if (display) {
        plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, "fpfh",100);

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(model, "model");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "model");
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::PointNormal>(model, normals, 10, 0.05, "normals");  // display every 1 points, and the scale of the arrow is 10
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();

        // Add point picking callback to viewer:
        struct callback_args cb_args;
        pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d(new pcl::PointCloud<pcl::PointXYZ>);
        cb_args.clicked_points_3d = clicked_points_3d;
        cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);
        viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);
        std::cout << "Shift + click on three floor points, then press 'Q'..." << std::endl;

//        viewer->spin();
//        cloud_mutex.unlock();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100); // Spin until 'Q' is pressed
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }

    return 0;
}*/
