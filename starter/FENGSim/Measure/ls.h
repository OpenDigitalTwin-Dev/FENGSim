#ifndef LS_H
#define LS_H

#include <math.h>
#include <vector>
#include <iostream>

class LS
{
private:
    int pc_num;
    std::vector<double> pc;
    double radius;
public:
    LS();
    void import ();
    double d (int i, double x, double y);
    double d_x (int i, double x, double y);
    double d_y (int i, double x, double y);
    double d_xx (int i, double x, double y);
    double d_yy (int i, double x, double y);
    void Newton ();
    void RotationX (double theta, double& x, double& y, double& z);
    void RotationZ (double theta, double& x, double& y, double& z);
};

#endif // LS_H
