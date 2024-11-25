// Use the light command to control the lightning
// Only one light source is possible, and it controls
// both hard shadows and specular and diffuse lightning.
set raytracer::light [0,0,5]

// Camera settings. Place these before first rule call.
set translation [-0.0593143 0.0238477 -20]
set rotation [-0.00666739 0.999819 0.0178465 -0.999778 -0.00630773 -0.0201334 -0.0200167 -0.0179761 0.999636]
set pivot [0 0 0]
set scale 0.379953

1 * { x 2.5  } 20 * {  rz 18 y 1  } 1 *  { z 1 s  0.2 0.2 2  color white } box
1 * { x 5  } 20 * {  rz 18 y 2 } 1 *  { z 1 s  0.2 0.2 2  color white } box
1 * { x 7.5 } 20 * {  rz 18 y 3 } 1 *  { z 1 s  0.2 0.2 2  color white } box
{ s 50 50  0.1 color white } box