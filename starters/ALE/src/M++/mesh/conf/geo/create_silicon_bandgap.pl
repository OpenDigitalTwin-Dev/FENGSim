open(Coordinate,"BandGapSiliconCoordinate.txt");
@coordinate=<Coordinate>;
close(Coordinate);

open(Number,"BandGapSiliconNumber.txt");
@number=<Number>;
close(Number);

open(Mesh,">BandGapSilicon3.geo");
print Mesh "POINTS\n";
for ($i=0; $i<scalar(@coordinate); $i++) {
    print Mesh $coordinate[$i];
}
print Mesh "CELLS\n";
for ($i=0; $i<scalar(@number); $i++) {
    print Mesh $number[$i];
}
close(Mesh);
