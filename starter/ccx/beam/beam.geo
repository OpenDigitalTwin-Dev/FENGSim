//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 5};
//+
Physical Surface("neumann", 13) = {6, 1, 3, 4, 2};
//+
Physical Surface("dirichlet", 14) = {5};
//+
Physical Volume("domain", 15) = {1};
