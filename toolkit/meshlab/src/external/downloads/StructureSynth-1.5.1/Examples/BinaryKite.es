set maxobjects 16000
10 * { y 1 } 10 * { z 1 }  1 * { a 0.8  sat 0.9  } r1 
set background #fff


rule r1   {
  { x 1  ry 4 } r1
  xbox
}

rule r1   {
{ x 1  ry -4  } r1
xbox
}

rule r1   {
{ x 1  rz -8  s 0.95 } r1
xbox
}

rule r1   {
{ x 1  rz 8  s 0.95   } r1
xbox
}



rule r2 maxdepth 36 {
{ ry 1  ry -13 x  1.2 b 0.99 h 12  } r2 
xbox
}

rule xbox {
  { s 1.1   color #000   } grid
  { b 0.7  color #000    }  box
}

rule xbox {
 { s 1.1   color #000     } grid
 { b 0.7  color #fff      } box
}
