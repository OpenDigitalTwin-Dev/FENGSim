//+
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.5, 0, 2*Pi};
//+
Line Loop(1) = {1};
//+
Surface(1) = {1};
