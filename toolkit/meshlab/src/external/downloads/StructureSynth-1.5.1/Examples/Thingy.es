#define steps 30
set minsize 0.01
set background white

10 * { s 0.8 rx 36 ry 10  } steps * { h 1 rz 360/steps } 1 * { x 10 rz 180} dspawn

rule dspawn {
   spawn
   m
}

rule m {
}

rule spawn {
  set seed initial
dbox
} 

rule dbox {
  { x 0.1 s 0.99 sat  0.99  } dbox
  box
}

rule dbox {
  { x 0.1 s 0.99 ry 1 } dbox
  box
}

rule dbox {
  { x 0.1 s 0.99 ry -1 } dbox
  box
}

rule dbox {
  { x 0.1 s 0.99 ry 1 } dbox
  box
}

rule dbox {
  { x 0.1 s 0.99 rz -3 } dbox
  box
}