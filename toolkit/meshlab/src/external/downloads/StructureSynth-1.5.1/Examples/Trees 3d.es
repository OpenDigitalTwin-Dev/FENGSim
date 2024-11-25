
set background white

{ h 30 sat 0.7 } seed
{ ry 180 h 30 sat 0.7 } seed
 
rule seed weight 100 { 
  box   
  { y 0.4 rx 1 s 0.995 b 0.995 } seed
}

rule seed weight 100 { 
  box   
  { y 0.4 rx 1 ry 1 s 0.995 b 0.995 } seed
}

rule seed weight 100 { 
  box   
  { y 0.4 rx 1 rz -1 s 0.995 b 0.995 } seed
}

rule seed weight 6 {  
  { rx 15 }  seed
  { ry 180 h 3  }  seed
} 
