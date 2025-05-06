SetFactory("OpenCASCADE");
Merge "EpplerE854.STEP";
//+
SetFactory("OpenCASCADE");
Disk(2) = {0.4, 0, 0, 5, 5};
//+
BooleanDifference{ Surface{2}; Delete; }{ Surface{1}; Delete; }
//+
MeshSize {2, 1} = 0.1;
//+
Physical Curve("farfield", 4) = {3};
//+
Physical Curve("airfoil", 5) = {1, 2};
