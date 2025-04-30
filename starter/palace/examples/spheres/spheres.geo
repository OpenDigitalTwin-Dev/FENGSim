//+
SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 5, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(2) = {-1, 0, 0, 0.4, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(3) = {1, 0, 0, 0.6, -Pi/2, Pi/2, 2*Pi};
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{3}; Delete; }
//+
Characteristic Length {9, 10, 5, 6} = 0.1;
//+
Physical Surface("outer", 2) = {4};
//+
Physical Surface("ball1", 3) = {5};
//+
Physical Surface("ball2", 4) = {3};
//+
Physical Volume("domain1", 1) = {1};
