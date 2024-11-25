/// The Nine Worthies
///  Syntopia 2009

set maxobjects 52200
set maxdepth 3000
set background #232
9 * { x 5 }  1 * { b 0.4  color #0a0    color white } r1 
// { s 1000 1000 1 color white } box 

// Camera settings. Place these before first rule call.
set translation [-4.50383 -2.80381 -20]
set rotation [0.993991 0.0947988 -0.0545559 0.0140638 0.383866 0.923278 0.108471 -0.918504 0.380229]
set pivot [0 0 0]
set scale 0.209524

rule r1 w  10  {
  { rz 15 h 1    y 1  h 4  z 0.01 s 0.999  } r1 
   r2
}

rule r1 w 0.1 {
  { rx 10    z 1  s 0.99 } r1 
   r2
}

rule r2 {
  box
  { s 1.1 0.2 0.2  color orange  } box
  grid
}


