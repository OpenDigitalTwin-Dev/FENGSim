#ifndef REGISTRATION_H
#define REGISTRATION_H

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/registration/icp.h"
#include "pcl/registration/icp_nl.h"

#include "MeasureDockWidget.h"

class Registration
{
    double* tran = new double[6];
    double* boxfit = new double[3];
public:
    Registration();
    void ICP (double p1=100, double p2=1e-8, double p3=1e-8);
    double* multiply (double a[], double b[]);
    void initial ();
    void copy ();
    void move ();
    void box_fit ();
    void SetTran (int i, double t)
    {
        tran[i] = t;
    }
    double GetTran (int i)
    {
        return tran[i];
    }
    MeasureDockWidget* dock;
    void SetDockWidget (MeasureDockWidget* _dock)
    {
        dock = _dock;
    }
};

#endif // REGISTRATION_H
