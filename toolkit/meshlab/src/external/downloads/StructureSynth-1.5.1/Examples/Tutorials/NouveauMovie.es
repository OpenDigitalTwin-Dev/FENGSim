// This system is meant to used
// by the 'JavaScript - Movie.es' example.

#define shrink s 0.996
#define _rz 0
#define _dofa 0.245
#define _dofb 0.09
#define _md 1000
set seed 14

// Camera settings. Place these before first rule call.
set translation [-1.54217 -1.76221 -20]
set rotation [0.530172 -0.847845 -0.00877037 0.100004 0.0522555 0.993614 -0.841972 -0.527663 0.112492]
set pivot [0 0 0]
set scale 1.13904

set raytracer::dof [_dofa,_dofb]

set maxdepth _md
set background #fff
1 * { rz _rz } 16 * { rz 20 color white } hbox

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