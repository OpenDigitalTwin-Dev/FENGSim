set maxobjects 52222
set background #a00
R1
{ x 3 s 0.9  } R1
{ y 3 s 0.7 } R1
{ y -3 s 0.6 } R1
{ x -3 s 0.5 } R1

// Camera settings. Place these before first rule call.
set translation [-2.59764 -5.03937 -20]
set rotation [0.946021 0.279788 0.163595 -0.166082 -0.0149648 0.985998 0.278318 -0.959945 0.0323107]
set pivot [0 0 0]
set scale 1.1874


rule R1 maxdepth 30 > endrule {
  { z 1 ry 6   s 0.91  hue 5  y 0.01 } R1
 mesh
}

rule R1 w 0.2 {
{ z 1 ry 6   s 0.99  hue 1 } R1
{ rz 90  } R1
mesh
}

rule endrule {
{ s 3 } sphere
}


