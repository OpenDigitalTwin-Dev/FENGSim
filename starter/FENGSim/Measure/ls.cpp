#include "ls.h"
#include <time.h>
#include <fstream>

LS::LS() {}

void LS::import()
{
    pc_num = 100;
    radius = 1;
    pc.resize(pc_num*2);
    std::ofstream out;
    out.open(std::string("/home/jiping/pc.dat").c_str());
    double dev;
    srand (time(NULL));
    for (int i = 0; i< pc_num; i++)
    {
        dev = ((double)rand()) / RAND_MAX;
        pc[i*2] = cos(2 * 3.1415926 * i / pc_num) + dev / 10;
        dev = ((double)rand()) / RAND_MAX;
        pc[i*2+1] = sin(2 * 3.1415926 * i / pc_num) + dev / 10;
        out << pc[i*2] << " " << pc[i*2+1] << std::endl;
    }
}

double LS::d(int i, double x, double y)
{
    return sqrt(pow(pc[i*2] - x, 2) + pow(pc[i*2+1] - y, 2));
}

double LS::d_x(int i, double x, double y)
{
    return 1.0 / 2.0 / d(i, x, y) * 2 * (x - pc[i*2]);
}

double LS::d_y(int i, double x, double y)
{
    return 1.0 / 2.0 / d(i, x, y) * 2 * (y - pc[i*2+1]);
}

double LS::d_xx(int i, double x, double y)
{
    return -1.0 / 2.0 / pow(d(i, x, y), 3) * 2 * (x - pc[i*2]);
}

double LS::d_yy(int i, double x, double y)
{
    return -1.0 / 2.0 / pow(d(i, x, y), 3) * 2 * (y - pc[i*2+1]);
}

void LS::Newton()
{
    double x_0 = 0;
    double y_0 = 0;
    double x_1 = 0;
    double y_1 = 0;
    double a11 = 0;
    double a12 = 0;
    double a21 = 0;
    double a22 = 0;
    double b1 = 0;
    double b2 = 0;
    for (int i = 0; i < 10000; i++)
    {
        for (int j = 0; j < pc_num; j++)
        {
            b1 += (d(j, x_0, y_0) - radius) * (x_0 - pc[2*j]) / d(j, x_0, y_0);
            b2 += (d(j, x_0, y_0) - radius) * (y_0 - pc[2*j+1]) / d(j, x_0, y_0);

            a11 += (d_x(j, x_0, y_0) / d(j, x_0, y_0) + (d(j, x_0, y_0) - radius) * d_xx(j, x_0, y_0)) * (x_0 - pc[2*j])
                    + (d(j, x_0, y_0) - radius) / d(j, x_0, y_0);
            a22 += (d_y(j, x_0, y_0) / d(j, x_0, y_0) + (d(j, x_0, y_0) - radius) * d_yy(j, x_0, y_0)) * (y_0 - pc[2*j+1])
                    + (d(j, x_0, y_0) - radius) / d(j, x_0, y_0);
            a12 += (d_y(j, x_0, y_0) / d(j, x_0, y_0) + (d(j, x_0, y_0) - radius) * d_yy(j, x_0, y_0)) * (x_0 - pc[2*j]);
            a21 += (d_x(j, x_0, y_0) / d(j, x_0, y_0) + (d(j, x_0, y_0) - radius) * d_xx(j, x_0, y_0)) * (y_0 - pc[2*j+1]);
        }
        x_1 = -1.0 / (a11 * a22 - a12 * a21) * (a22 * b1 - a12 * b2);
        y_1 = -1.0 / (a11 * a22 - a12 * a21) * (-a21 * b1 + a11 * b2);
        x_0 += x_1;
        y_0 += y_1;
        //std::cout << a11 << " " << a12 << " " << a22 << " " << a21 << " " << b1 << " " << b2 << std::endl;
        a11 = a12 = a21 = a22 = b1 = b2 = 0;
        std::cout << "newton step " << i << ": " << x_0 << " " << y_0 << std::endl;
    }
    std::ofstream out;
    out.open(std::string("/home/jiping/circle.dat").c_str());
    out << x_0 << " " << y_0 << " " << 1 << std::endl;
}

void LS::RotationX (double theta, double& x, double& y, double& z)
{
    theta = theta / 180 * 3.1415926;
    double A[3][3];
    A[0][0] = 1;
    A[0][1] = 0;
    A[0][2] = 0;
    A[1][0] = 0;
    A[1][1] = cos(theta);
    A[1][2] = sin(theta);
    A[2][0] = 0;
    A[2][1] = -sin(theta);
    A[2][2] = cos(theta);

    double x1 = A[0][0] * x + A[0][1] * y + A[0][2] * z;
    double y1 = A[1][0] * x + A[1][1] * y + A[1][2] * z;
    double z1 = A[2][0] * x + A[2][1] * y + A[2][2] * z;

    x = x1;
    y = y1;
    z = z1;
}

void LS::RotationZ (double theta, double& x, double& y, double& z)
{
    theta = theta / 180 * 3.1415926;
    double A[3][3];
    A[0][0] = cos(theta);
    A[0][1] = sin(theta);
    A[0][2] = 0;
    A[1][0] = -sin(theta);
    A[1][1] = cos(theta);
    A[1][2] = 0;
    A[2][0] = 0;
    A[2][1] = 0;
    A[2][2] = 1;

    double x1 = A[0][0] * x + A[0][1] * y + A[0][2] * z;
    double y1 = A[1][0] * x + A[1][1] * y + A[1][2] * z;
    double z1 = A[2][0] * x + A[2][1] * y + A[2][2] * z;

    x = x1;
    y = y1;
    z = z1;
}
