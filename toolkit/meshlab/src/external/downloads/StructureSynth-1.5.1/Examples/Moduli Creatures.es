// Moduli Creatures. Syntopia 2010.
// This one works best when raytraced in e.g. Sunflow.

// Camera settings. Place these before first rule call.
set translation [-8.07939 -6.42262 -20]
set rotation [-0.608586 0.78168 -0.136359 0.78734 0.616241 0.0185897 0.0985613 -0.0960483 -0.99048]
set pivot [0 0 0]
set scale 0.625595

set background grey
50 * { x 1 rz 5 s 0.99 color red } 20 * { y 1 rz 3 hue 1 } d

rule d md 30 {
{ z 1 rx 5 } d
{ s 0.01 1 1.2 sat 0.4 } box 
} 

rule d w 0.1 {
 e 
} 

rule e md 5 {
{ z 1 rx 5 } e
{ s 0.01  1 1.2 sat 0.4 } box 
}