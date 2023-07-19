open(Coordinate,"BandGapAirCoordinate.txt");
@coordinate=<Coordinate>;
close(Coordinate);

open(Number,"BandGapAirNumber.txt");
@number=<Number>;
close(Number);

open(Mesh,">BandGapAir3.geo");
print Mesh "POINTS\n";
for ($i=0; $i<scalar(@coordinate); $i++) {
    print Mesh $coordinate[$i];
}
print Mesh "CELLS\n";
for ($i=0; $i<scalar(@number); $i++) {
    print Mesh $number[$i];
}
close(Mesh);
