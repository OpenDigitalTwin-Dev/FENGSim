// ------ Camera settings.
set translation [0 -0.367113 -20]
set rotation [0.963994 0.0575931 -0.259613 0.0514549 0.917418 0.394584 0.260899 -0.393735 0.881422]
set pivot [0 0 0]
set scale 0.408795

// ------ The actual EisenScript
set background #fff
set maxdepth 400

 r0

rule r0 {
  3 * { rz 120  } R1
  3 * { rz 120 } R2
}

rule R1 {
  { x 1.3 rx 1.57 rz 6 ry 3 s 0.99 hue 0.4 sat 0.99  } R1
  { s 4 } sphere::shiny
}

rule R2 {
  { x -1.3 rz 6 ry 3 s 0.99 hue 0.4 sat 0.99 } R2
  { s 4 }  sphere
}

// ----- Settings for internal raytracer

set raytracer::shadows false

// the number of samples controls the quality
// '6' means 6x6 samples per pixels, and is the default.
set raytracer::samples 6

// dof is depth-of-field.
// Use 'Edit | Show 3D Object Information' to find the correct plane 
// comment the line below to disable this.
set raytracer::dof [0.23,0.07]

// Set materials either globally,
// or for a selected tag (e.g. 'shiny')
set raytracer::shiny::reflection 0.3
set raytracer::phong [0.5,0.6,0.2]
 