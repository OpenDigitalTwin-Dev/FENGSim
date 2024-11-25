/*
	This is just a simple demonstration
      of the various primitives in Structure Synth.
*/

set maxdepth 20
set background white

// Camera settings. Place these before first rule call.
set translation [4.6096 -1.3831 -20]
set rotation [-0.923481 0.370221 -0.100283 -0.119863 -0.0301876 0.992333 0.364361 0.928431 0.0722537]
set pivot [0 0 0]
set scale 0.38721

R6
{ x 4 } R2
{ x 8 } R3
{ x 12 } R4
{ x 16 } R5
{ x 20 } R1

rule R1 maxdepth 8 {
  { z 1 rz 0 rx 10 s 1.2 h 20 } R1
  sphere
}


rule R2 maxdepth 8 {
{ z 1 rz 0 rx 10 s 1.2 h 20 } R2
box
}

rule R3  maxdepth 8  {
{ z 1 rz 0 rx 10 s 1.2 h 20 } R3
dot
}

rule R4  maxdepth 8  {
{ z 1 rz 0 rx 10 s 1.2 h 20 } R4
line
}

rule R5  maxdepth 8  {
{ z 1 rz 0 rx 10 s 1.2 h 20 } R5
grid
}

rule R6   maxdepth 8  {
{ z 1 rz 0 rx 10 s 1.2 h 20 } R6
mesh
}