// Demonstration of Blend
//
// The blend operator blends the current color 
// with the specified color. 
//
// Notice that colors are blended in HSV space,
// explaining why blending from red to blue will 
// go through green and yellow.


set maxdepth 400
R1
R2

rule R1 {
  { x 1 rz 6 ry 6 s 0.99 blend blue 0.04 } R1
  { s 2 } sphere
}

rule R2 {
  { x -1 rz 6 ry 6 s 0.99 blend green 0.04 } R2
  { s 2 } sphere
} 