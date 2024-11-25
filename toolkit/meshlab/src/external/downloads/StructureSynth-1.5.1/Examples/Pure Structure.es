#define shrink  s 0.996

set maxdepth 1000
set background white 
set syncrandom true 

//set colorpool image:001.jpg

set colorpool list:white,white,white,orange
set minsize 0.5

5 * { x 2 } 1 * { s 0.2 0.2 1 } s2

rule s2 md 25 {
   { z 1  } s2
   start
}



rule start { 6  * { rz 60 } hbox }

rule hbox md 10 { r }

rule r {
 set seed initial 
 forward
}

rule r {  turn }
rule r  {  turn2 }

rule forward md 90  >  r  {
  dbox
{ rz 2 x 0.1 shrink } forward
}

rule turn md 90  >  r  {
    dbox
{ rz 2  x 0.1 shrink } turn
}

rule turn2 md 90  >  r  {
  dbox
{ rz -2  x 0.1 shrink } turn2
}

rule turn3 md 90  >  r  {
    dbox
{ ry -2 x 0.1 shrink } turn3
}

rule turn4 md 90  >  r  {
  dbox
{ ry -2 x 0.1 shrink } turn4
}

rule turn5 md 90  >  r  {
    dbox
{ rx -2 x 0.1 shrink } turn5
}

rule turn6 md 90  >  r  {
  dbox
{ rx -2 x 0.1 shrink } turn6
}

rule dbox {
{ s 0.2 1 1  color random } box
}