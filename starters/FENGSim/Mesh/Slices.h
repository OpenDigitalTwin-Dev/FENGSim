#ifndef SLICES_H
#define SLICES_H

#include <vector>
#include <iostream>

class Slices
{
    std::vector<int> slices_num;
    std::vector<double> height;
    std::vector<double> xy;
    double convex_hull[6];
public:
    Slices();
    void Clear () {
        slices_num.clear();
        height.clear();
        xy.clear();
    }
    void InsertSlice (double h) {
        slices_num.push_back(0);
        height.push_back(h);
    }
    void InsertXY (double x, double y) {
        xy.push_back(x);
        xy.push_back(y);
        int n = slices_num.size();
        slices_num[n-1] = slices_num[n-1] + 1;
    }
    int size ()
    {
        return slices_num.size();
    }
    int SlicesNum (int n)
    {
        return slices_num[n];
    }
    double SliceHeight (int n)
    {
        return height[n];
    }
    double X (int n, int m)
    {
        int l = 0;
        for (int i = 0; i < n; i++)
        {
            l += slices_num[i];
        }
        l += m;
        return xy[l*2];
    }
    double Y (int n, int m)
    {
        int l = 0;
        for (int i = 0; i < n; i++)
        {
            l += slices_num[i];
        }
        l += m;
        return xy[l*2+1];
    }
    void SetConvexHull ()
    {
        double x_max = -1e10;
        double x_min = 1e10;
        double y_max = -1e10;
        double y_min = 1e10;
        double z_max = -1e10;
        double z_min = 1e10;
        int n = 0;
        for (int i = 0; i < slices_num.size(); i++) {
            for (int j = 0; j < slices_num[i]; j++) {
                double x = xy[(n+j)*2];
                double y = xy[(n+j)*2+1];
                double z = height[i];
                if (x > x_max)
                    x_max = x;
                if (x < x_min)
                    x_min = x;
                if (y > y_max)
                    y_max = y;
                if (y < y_min)
                    y_min = y;
                if (z > z_max)
                    z_max = z;
                if (z < z_min)
                    z_min = z;
            }
            n += slices_num[i];
        }
        convex_hull[0] = x_min;
        convex_hull[1] = x_max;
        convex_hull[2] = y_min;
        convex_hull[3] = y_max;
        convex_hull[4] = z_min;
        convex_hull[5] = z_max;
        std::cout << x_min << " " << x_max << " " << y_min << " " << y_max << " " << z_min << " " << z_max << std::endl;
    }
    double x_min () {
        return convex_hull[0];
    }
    double x_max () {
        return convex_hull[1];
    }
    double y_min () {
        return convex_hull[2];
    }
    double y_max () {
        return convex_hull[3];
    }
    double z_min () {
        return convex_hull[4];
    }
    double z_max () {
        return convex_hull[5];
    }
};

#endif // SLICES_H
