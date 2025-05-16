//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Physical Surface("current", 13) = {6};
//+
Physical Surface("other", 14) = {4, 2, 3, 1, 5};
//+
Physical Volume("domain", 15) = {1};
