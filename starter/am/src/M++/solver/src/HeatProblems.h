// file: Solution.h
// author: Jiping Xin

#include "m++.h"

class HeatProblems {
    int example_id;
public:
    HeatProblems () {
        ReadConfig(Settings, "EXAMPLE", example_id);
    }
    void SetSubDomain (Mesh& M);
    void SetBoundaryType (Mesh& M);
    bool IsDirichlet (int id);
    double f (Point p, double t) const;
    double g_D (Point p, int id, double t) const;
    double g_N (Point p, int id, double t) const;
    double v (const Point& p) const;
    double u (const Point& p, double t) const;
};
