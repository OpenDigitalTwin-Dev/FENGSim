#include "Registration.h"
#include "fstream"
#include "ui_MeasureDockWidget.h"

Registration::Registration()
{
        for (int i = 0; i < 6; i++)
                tran[i] = 0;
}

void Registration::ICP (double p1, double p2, double p3)
//void ICP (pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_target,
//        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr &output )
{
        // Due to the multi-threaded nature of PCL,
        // the standard pipe cout and cerr away wont work.
        // Instead you need to use the in-built functions provided by PCL to turn off the console printing.
        // pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>);

        // fill in the source data
        std::ifstream is;
        is.open("./data/meas/fengsim_meas_cloud_source2.vtk");
        const int len = 256;
        char L[len];
        int n1 = 0;
        while (is.getline(L,len))
        {
                n1++;
        }
        is.close();

        // fill in the source data
        cloud_source->width    = n1;
        cloud_source->height   = 1;
        cloud_source->is_dense = false;
        cloud_source->points.resize (cloud_source->width * cloud_source->height);
        is.open("./data/meas/fengsim_meas_cloud_source2.vtk");
        int j = 0;
        double tol1 = 0;
        while (is.getline(L,len))
        {
                double z[3];
                sscanf(L,"%lf %lf %lf", z, z+1, z+2);
                cloud_source->points[j].x = z[0];
                cloud_source->points[j].y = z[1];
                cloud_source->points[j].z = z[2];
                double tol2 = abs(sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]) - 5);
                if (tol2 > tol1) tol1 = tol2;
                j++;
        }
        is.close();
        std::cout <<"icp source tol: " << tol1 << std::endl;

        // fill in the target data
        is.open("./data/meas/fengsim_meas_cloud_target.vtk");
        int n2 = 0;
        while (is.getline(L,len))
        {
                n2++;
        }
        is.close();
        cloud_target->width    = n2;
        cloud_target->height   = 1;
        cloud_target->is_dense = false;
        cloud_target->points.resize (cloud_target->width * cloud_target->height);
        is.open("./data/meas/fengsim_meas_cloud_target.vtk");
        j = 0;
        tol1 = 0;
        while (is.getline(L,len))
        {
                double z[3];
                sscanf(L,"%lf %lf %lf", z, z+1, z+2);
                cloud_target->points[j].x = z[0];
                cloud_target->points[j].y = z[1];
                cloud_target->points[j].z = z[2];
                double tol2 = abs(sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2])-5);
                if (tol2 > tol1) tol1 = tol2;
                j++;
        }
        is.close();
        std::cout <<"icp target tol: " << tol1 << std::endl;

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ, double> icp;

        icp.setInputSource(cloud_source);
        icp.setInputTarget(cloud_target);

        std::cout << "icp parameter: " << p1 << " " << p2 << " " << p3 << std::endl;


        icp.setMaximumIterations(1000);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1e-8);


        //icp.setMaxCorrespondenceDistance(1e-5);
        //icp.setRANSACOutlierRejectionThreshold(1e-5);

        pcl::PointCloud<pcl::PointXYZ> Final;
        icp.align(Final);
        std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                     icp.getFitnessScore() << " fitness epsilon: " << icp.getEuclideanFitnessEpsilon() <<
                     " transform epsilon: " << icp.getTransformationEpsilon() << std::endl;


        std::cout << icp.getFinalTransformation() << std::endl;

        //dock->ui->textEdit->append(QString("has converged:") + QString::number(icp.hasConverged()) + QString("\r\n"));
        //dock->ui->textEdit->append(QString("score:") + QString::number(icp.getFitnessScore()) + QString("\r\n"));
        //dock->ui->textEdit->append(QString("fitness epsilon: ") + QString::number(icp.getFitnessScore()) + QString("\r\n"));

        std::ofstream out;
        out.open(std::string("./data/meas/fengsim_meas_icp_final.vtk").c_str());
        tol1 = 0;
        for (int i = 0; i < Final.size(); i++)
        {
                out << std::setprecision(16)
                       //out
                    << (Final.points)[i].x << " "
                    << (Final.points)[i].y << " "
                    << (Final.points)[i].z << std::endl;

                double z[3];
                z[0] = (Final.points)[i].x;
                z[1] = (Final.points)[i].y;
                z[2] = (Final.points)[i].z;
                double tol2 = abs(sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2])-5);
                if (tol2 > tol1) tol1 = tol2;
        }
        out.close();
        std::cout <<"icp final tol: " << tol1 << std::endl;
}

void Registration::initial() {
        char ch;
        std::ifstream is;
        is.open("./data/cloud_source.vtk");
        std::ofstream out;
        out.open("./data/cloud_source_move.vtk");
        while (is.get(ch)) {
                out << ch;
        }
}

void Registration::copy() {
        char ch;
        std::ifstream is;
        is.open("./data/cloud_source.vtk");
        std::ofstream out;
        out.open("./data/cloud_source_move.vtk");
        while (is.get(ch)) {
                out << ch;
        }
}

double* Registration::multiply (double a[], double b[]) {
        double* t = new double[4];
        for (int i = 0; i < 4; i++) {
                t[i] = 0;
                for (int j = 0; j < 4; j++) {
                        t[i] += a[i*4+j] * b[j];
                }
        }
        return t;
}

void Registration::move () {
        double a[16];
        a[0] = 1;
        a[1] = 0;
        a[2] = 0;
        a[3] = tran[0] + boxfit[0];

        a[4] = 0;
        a[5] = 1;
        a[6] = 0;
        a[7] = tran[1] + boxfit[1];

        a[8] = 0;
        a[9] = 0;
        a[10] = 1;
        a[11] = tran[2] + boxfit[2];

        a[12] = 0;
        a[13] = 0;
        a[14] = 0;
        a[15] = 1;

        double a1[16];
        a1[0] = 1;
        a1[1] = 0;
        a1[2] = 0;
        a1[3] = 0;

        a1[4] = 0;
        a1[5] = cos(tran[3]/180*3.1415926);
        a1[6] = sin(tran[3]/180*3.1415926);
        a1[7] = 0;

        a1[8] = 0;
        a1[9] = -sin(tran[3]/180*3.1415926);
        a1[10] = cos(tran[3]/180*3.1415926);
        a1[11] = 0;

        a1[12] = 0;
        a1[13] = 0;
        a1[14] = 0;
        a1[15] = 1;

        double a2[16];
        a2[0] = cos(tran[4]/180*3.1415926);
        a2[1] = 0;
        a2[2] = -sin(tran[4]/180*3.1415926);
        a2[3] = 0;

        a2[4] = 0;
        a2[5] = 1;
        a2[6] = 0;
        a2[7] = 0;

        a2[8] = sin(tran[4]/180*3.1415926);
        a2[9] = 0;
        a2[10] = cos(tran[4]/180*3.1415926);
        a2[11] = 0;

        a2[12] = 0;
        a2[13] = 0;
        a2[14] = 0;
        a2[15] = 1;

        double a3[16];
        a3[0] = cos(tran[5]/180*3.1415926);
        a3[1] = sin(tran[5]/180*3.1415926);
        a3[2] = 0;
        a3[3] = 0;

        a3[4] = -sin(tran[5]/180*3.1415926);
        a3[5] = cos(tran[5]/180*3.1415926);
        a3[6] = 0;
        a3[7] = 0;

        a3[8] = 0;
        a3[9] = 0;
        a3[10] = 1;
        a3[11] = 0;

        a3[12] = 0;
        a3[13] = 0;
        a3[14] = 0;
        a3[15] = 1;

        std::ifstream is;
        is.open("./data/meas/fengsim_meas_cloud_source.vtk");
        std::ofstream out;
        out.open("./data/meas/fengsim_meas_cloud_source2.vtk");
        const int len = 256;
        char L[len];
        while (is.getline(L,len))
        {
                double z[4];
                sscanf(L,"%lf %lf %lf", z, z+1, z+2);
                z[3] = 1;
                double* t = multiply(a,z);
                t = multiply(a1,t);
                t = multiply(a2,t);
                t = multiply(a3,t);
                out << t[0] << " " << t[1] << " " << t[2] << std::endl;
        }
}

void Registration::box_fit () {
        double x1[6];
        x1[0] = -1e20;
        x1[1] = 1e20;
        x1[2] = -1e20;
        x1[3] = 1e20;
        x1[4] = -1e20;
        x1[5] = 1e20;
        double x2[6];
        x2[0] = -1e20;
        x2[1] = 1e20;
        x2[2] = -1e20;
        x2[3] = 1e20;
        x2[4] = -1e20;
        x2[5] = 1e20;
        std::ifstream is;
        const int len = 256;
        char L[len];
        is.open("./data/cloud_source.vtk");
        while (is.getline(L,len))
        {
                double z[3];
                sscanf(L,"%lf %lf %lf", z, z+1, z+2);
                if (x1[0] < z[0])
                {
                        x1[0] = z[0];
                }
                if (x1[1] > z[0])
                {
                        x1[1] = z[0];
                }
                if (x1[2] < z[1])
                {
                        x1[2] = z[1];
                }
                if (x1[3] > z[1])
                {
                        x1[3] = z[1];
                }
                if (x1[4] < z[2])
                {
                        x1[4] = z[2];
                }
                if (x1[5] > z[2])
                {
                        x1[5] = z[2];
                }
        }
        is.close();
        is.open("./data/cloud_target.vtk");
        while (is.getline(L,len))
        {
                double z[3];
                sscanf(L,"%lf %lf %lf", z, z+1, z+2);
                if (x2[0] < z[0])
                {
                        x2[0] = z[0];
                }
                if (x2[1] > z[0])
                {
                        x2[1] = z[0];
                }
                if (x2[2] < z[1])
                {
                        x2[2] = z[1];
                }
                if (x2[3] > z[1])
                {
                        x2[3] = z[1];
                }
                if (x2[4] < z[2])
                {
                        x2[4] = z[2];
                }
                if (x2[5] > z[2])
                {
                        x2[5] = z[2];
                }
        }
        boxfit[0] = (x2[0] + x2[1]) / 2 - (x1[0] + x1[1]) / 2;
        boxfit[1] = (x2[2] + x2[3]) / 2 - (x1[2] + x1[3]) / 2;
        boxfit[2] = (x2[4] + x2[5]) / 2 - (x1[4] + x1[5]) / 2;
        std::cout << x1[0] << " " << x1[1] << " " << x1[2] << " " << x1[3] << " " << x1[4] << " " << x1[5] << std::endl;
        std::cout << x2[0] << " " << x2[1] << " " << x2[2] << " " << x2[3] << " " << x2[4] << " " << x2[5] << std::endl;
}
