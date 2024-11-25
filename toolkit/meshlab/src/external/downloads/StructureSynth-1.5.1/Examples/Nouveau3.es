// Nouveau variant (Rounded tubes)

#define shrink s 0.996


set maxdepth 1000
set background #fff
10 * { x 0 color white ry 10  } 2 * { rz 180  } hbox

rule hbox { r}
rule r { forward }
rule r { turn }
rule r { turn2 }
rule r { turn4 }
rule r { turn3 }
rule r { turn5 }
rule r { turn6 }

rule forward md 90 > r {
dbox
{ rz 2 x 0.1 shrink } forward
}

rule turn md 90 > r {
dbox
{ rz 2 x 0.1 shrink } turn
}

rule turn2 md 90 > r {
dbox
{ rz -2 x 0.1 shrink } turn2
}

rule turn3 md 90 > r {
dbox
{ ry -2 x 0.1 shrink } turn3
}

rule turn4 md 90 > r {
dbox
{ ry -2 x 0.1 shrink } turn4
}

rule turn5 md 90 > r {
dbox
{ ry -2 x 0.1 shrink } turn5
}

rule turn6 md 90 >  r {
dbox
{ ry -2 x 0.1 shrink } turn6
}

rule dbox {
{ s 0.2 1 1 } sphere
}