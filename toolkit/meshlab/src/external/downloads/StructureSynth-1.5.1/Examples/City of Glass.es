set maxobjects 117600
{ a 0.4 sat 0.5 } grinder 

rule grinder { 
  36 * { ry 10 z 1.2 b 0.99 h 2  } arm 
} 

rule arm {
 xbox
 {  rz -1 rx 3 z 0.22 x 0.3 ry 10 s 0.998  x 0.5 sat 0.9995 hue 0.05 }  arm
}

rule arm {
   xbox
  {  rz 1 rx 3 z 0.22 x 0.3 ry 10 s 0.99  x -0.5 sat 0.9995 hue 0.15 }  arm
}

rule xbox {
  box
}

rule xbox {
 { s 0.9 } grid
 { b 0.7   } box
}
