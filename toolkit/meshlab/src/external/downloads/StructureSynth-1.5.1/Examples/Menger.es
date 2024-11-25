R1

rule R1 maxdepth 3 > c2 {
	{ s 1/3 x -1 y -1  } R1 
	{ s 1/3 x -1 y -1  z -1 } R1 
	{ s 1/3 x -1 y -1  z +1 } R1 
	{ s 1/3 x 1 y -1  } R1 
	{ s 1/3 x 1 y -1  z -1 } R1 
	{ s 1/3 x 1 y -1  z +1 } R1 
	{ s 1/3  y -1  z -1 } R1  
	{ s 1/3  y -1  z +1 } R1 
	{ s 1/3 x -1 y 1  } R1 
	{ s 1/3 x -1 y 1  z -1 } R1 
	{ s 1/3 x -1 y 1  z +1 } R1 
	{ s 1/3 x 1 y 1  } R1 
	{ s 1/3 x 1 y 1  z -1 } R1 
	{ s 1/3 x 1 y 1  z +1 } R1 
	{ s 1/3  y 1  z -1 } R1  
	{ s 1/3  y 1  z +1 } R1 
	{ s 1/3 x -1   z -1 } R1 
	{ s 1/3 x -1   z +1 } R1 
	{ s 1/3 x 1    z -1 } R1 
	{ s 1/3 x 1    z +1 } R1 
}

rule c2 {
 box
}




