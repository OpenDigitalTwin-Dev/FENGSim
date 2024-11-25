// Reflection.es

set raytracer::shadows false
set raytracer::reflection 0.2
set raytracer::phong [0.5,0.4,0.2]
set raytracer::max-depth 5
set background #fff
set maxdepth 400
2 * { rx 38 color white } R1

rule R1 {
  { x 1.3 rz 6 ry 3 s 0.99 hue 1 sat 0.99 blend red 0.02 } R1
  { s 6 a 1 } sphere 
  { s 7 hue 120 a 0.2 } sphere 
}

rule R2 {
  { x -1.3 rz 6 ry 3 s 0.99 hue 1 sat 0.99 blend red 0.02 } R2
  { s 6 a 1 } sphere
  { s 7 hue 120 a 0.2 } sphere 
}