/*
	An example of using GUI preprocessor defines.

       The "float" suffix makes it possible to control variables via sliders.
       The 'int' suffix is identical to the above, but only accepts natural numbers.
*/

#define sizeStep 0.94 (float:0-1)
#define angle1 20 (float:0-90)
#define angle2 6 (float:0-90)
#define iterations 6 (int:1-90)


set maxdepth 100
set background black

iterations * { rx 10  x 0.2 sat 0.95  } R


rule R { 
set seed initial 
R1 
}

rule R1 {
  {  x 0.6 rz 0 ry angle1 s sizeStep hue 1 a 0.99 } R1
  { s 1 1  0.1  color green  } sbox
}

rule R1 {
  {  x 0.6 rz angle2 ry 0 s sizeStep hue 1 a 0.99 } R1
  { s 1 1  0.1  color red } sbox
}

rule sbox {
 { color black } grid
  {  b 0.8 hue 67 } box
}
