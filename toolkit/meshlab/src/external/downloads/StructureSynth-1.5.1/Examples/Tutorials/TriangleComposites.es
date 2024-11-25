// This example shows how to build
// custom primitives from individual polygons.
// Note that the Triangle primitives are NOT supported
// by most of the templates.

5 * { x 2 } 5 * { y 2 hue 20 } house

rule house {
	{ color black s 1.01 } grid
       box
 	{ z 1 sat 0.4 } pyramid
}

rule house {
	{ color black s 1.01 } grid
	 box
	{ z 1 } bar
}

rule pyramid { 
	 triangle[0,0,0;1,0,0;0.5,0.5,0.5] 
	 triangle[1,0,0;1,1,0;0.5,0.5,0.5] 
	 triangle[1,1,0;0,1,0;0.5,0.5,0.5] 
	 triangle[0,1,0;0,0,0;0.5,0.5,0.5] 
} 

rule bar { 
	 triangle[0,0,0;1,0,0;0.5,0,0.5] 
	 triangle[1,0,0;1,1,0;0.5,0,0.5] 
	 triangle[0.5,0,0.5;1,1,0;0.5,1,0.5] 
	 triangle[0,1,0;0.5,1,0.5;1,1,0] 
	 triangle[0,0,0;0.5,0,0.5;0,1,0] 
 	 triangle[0.5,0,0.5;0.5,1,0.5;0,1,0] 
} 