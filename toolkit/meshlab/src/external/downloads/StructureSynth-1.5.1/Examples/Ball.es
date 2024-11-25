set maxdepth 2000
{ a 0.9 hue 30 } R1 

rule R1 w 10 { 
{ x 1  rz 3 ry 5  } R1
{ s 1 1 0.1 sat 0.9 } box
} 

rule R1 w 10 { 
{ x 1  rz -3 ry 5  } R1
{ s 1 1 0.1 } box
} 

