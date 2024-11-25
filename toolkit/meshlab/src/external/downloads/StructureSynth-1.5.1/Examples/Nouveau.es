#define shrink s 0.996

set seed 9
// Camera settings. Place these before first rule call.
set translation [-2.66914 -1.23905 -20]
set rotation [0.707436 -0.702911 -0.0738339 0.542177 0.472696 0.694696 -0.453408 -0.531484 0.715504]
set pivot [0 0 0]
set scale 0.953754

set raytracer::dof [0.245,0.09]

set maxdepth 1000
set background #fff
16 * { rz 20 color white } hbox

rule hbox { r}
rule r { forward }
rule r { turn }
rule r { turn2 }
rule r { turn4 }
rule r { turn3 }
//rule r { turn5 }
//rule r { turn6 }

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
{ rx -2 x 0.1 shrink } turn5
}

rule turn6 md 90 >  r {
dbox
{ rx -2 x 0.1 shrink } turn6
}

rule dbox {
{ s 0.2 1 1 } box
}