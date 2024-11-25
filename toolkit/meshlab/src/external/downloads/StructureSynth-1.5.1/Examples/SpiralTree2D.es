set maxdepth 600
{ h 30 sat 0.7 } spiral
{ ry 180 h 30 sat 0.7 } spiral
 
rule spiral w 100 { 
 box   
{ y 0.4 rx 1 s 0.995 b 0.995 } spiral
}

rule spiral w 1 {  
spiral
{ ry 180 h 3  }  spiral
} 
