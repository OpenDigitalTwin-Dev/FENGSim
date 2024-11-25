set background #000
#define _md 400
#define _rz 0
#define _zoom 1

set maxdepth _md

{ rz _rz s _zoom } r0

rule r0 {
3 * { rz 120  } R1
3 * { rz 120 } R2
}

rule R1 {
{ x -1.3 rz 1.57 rz 6 ry 13 s 0.99 hue 1 sat 0.99 } R1
{ s 4   color black } box
}

rule R2 {
{ x -1.3 rz 6 ry 13 s 0.99 hue 1 sat 0.99 } R2
{ s 4 a 1  color white } box
} 