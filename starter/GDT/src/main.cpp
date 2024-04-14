#include "sac_ia.h"
#include "icp.h"
#include <pcl/filters/uniform_sampling.h>

int main (int argc, char** argv) {  
    double uns_radius = 1;
    double sacia_itnum = 1000;
    std::cout << "uniform sampling radius: " << uns_radius << std::endl;
    std::cout << "sacia iterative num: " << sacia_itnum << std::endl;

    PointCloud<PointXYZ> _cloud_source, cloud_source,
                         _cloud_target, cloud_target,
                         cloud_sacia, cloud_icp;
    import_mesh2pc("./data/mesh/fengsim_mesh.vtk", _cloud_target);
    import_pc("./data/meas/fengsim_meas_source.vtk", _cloud_source);
    std::cout << "source num: " << _cloud_source.size() << std::endl;
    std::cout << "target num: " << _cloud_target.size() << std::endl;

    // #################################################
    // ############ step 1 uniform sampling ############
    // #################################################
    std::cout << "**uniform sampling**" << std::endl;
    UniformSampling<PointXYZ> uniform;
    uniform.setRadiusSearch(uns_radius);  // 1m

    uniform.setInputCloud(_cloud_source.makeShared ());
    uniform.filter(cloud_source);

    uniform.setInputCloud(_cloud_target.makeShared ());
    uniform.filter(cloud_target);

    //export_pc_to_vtk(cloud_source, "./data/meas/fengsim_meas_source_us.vtk");
    //export_pc_to_vtk(cloud_target, "./data/meas/fengsim_meas_target_us.vtk");
    std::cout << "source num: " << cloud_source.size() << std::endl;
    std::cout << "target num: " << cloud_target.size() << std::endl;
    
    for (int i=1; i<6; i++) {
        // #################################################
        // ############ step 2 sacia            ############
        // #################################################
        std::cout << "**sacia**" << std::endl;
	sacia_align sacia;
	sacia.align(cloud_target, cloud_source, cloud_sacia, uns_radius, i*sacia_itnum);

	// #################################################
        // ############ step 3 icp              ############
        // #################################################
	std::cout << "**icp**" << std::endl;
	icp_align icp;
	double fit = icp.align(cloud_sacia, _cloud_source, cloud_icp);
	std::cout << fit << std::endl;

	if (fit < 1) {
	    std::cout << "**registration done.**" << std::endl;
	    Eigen::Matrix4f transform = sacia.transform.inverse()*icp.transform.inverse();
	    transformPointCloud (cloud_source, cloud_icp, transform);
	    export_matrix(transform, "./data/meas/trans_matrix");
	    //export_pc_to_vtk(cloud_icp, "./data/meas/fengsim_meas_icp.vtk");
	    break;
	}
	cloud_sacia.clear();
	cloud_icp.clear();
    }
    return 0;
}
