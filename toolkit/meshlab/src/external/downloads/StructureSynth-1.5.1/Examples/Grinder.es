// Grinder.es

set maxobjects 117600
grinder 

rule grinder { 
  36 * { ry 10 z 1.2 b 0.99 } 1 * { ry 34  rx 34 }  arm 
} 

rule arm {
 xbox
 {  rz 2 rx 3 z 0.22 x 0.3 ry 10 s 0.999  x 0.5 sat 0.9995 hue 0.05 }  arm
}

rule arm {
   xbox
  {  rz 2 rx 3 z 0.22 x 0.3 ry 10 s 0.99  x 0.5 sat 0.9995 hue 0.15 }  arm
}

rule xbox {
  { b 0.7   } box
}
